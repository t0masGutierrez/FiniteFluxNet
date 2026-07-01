from __future__ import annotations
import jax
import jax.numpy as jnp
import jax.random as jr


def sample_task(examples: jax.Array, key: jax.Array, *, k: int) -> tuple[jax.Array, jax.Array]:
    """
    sample task from examples \\
    task{ \\
    context = examples[0], examples[10], examples[20], examples[30] \\
    query = examples[40] \\
    }

    parameters
    ----------
    examples: jax.Array
        input-target examples
    key: jax.Array
        random number generator
    k: int
        number of context examples

    returns
    -------
    context: jax.Array
        input-target examples
    query: jax.Array
        held-out input-target example
    """
    # TODO: parameterize the number of query examples
    indices = jr.choice(key, len(examples), shape=(k + 1,), replace=False)
    context = jnp.array([examples[i] for i in indices[:-1]])
    query = jnp.array(examples[indices[-1]])
    return context, query


def tokenize(x: jax.Array, context: jax.Array, query: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    convert function values into sequence values

    parameters
    ----------
    x: jax.Array
        spatial coordinates
    context: jax.Array
        input-target examples
    query: jax.Array
        held-out input-target example

    returns
    -------
    input_tokens: jax.Array
        sequence of tokenized input function values
    target_tokens: jax.Array
        sequence of tokenized target function values
    """
    input_tokens = list()
    for i in range(context.shape[0]):  # example index
        for j in range(context.shape[1]):  # input-target index
            for k in range(context.shape[2]):  # spatial index
                role = None
                if j == 0:
                    role = 0  # context input
                else:
                    role = 1  # context target
                x_k = x[k]
                tok = (x_k, context[i][j][k], role)
                input_tokens.append(tok)
    target_tokens = list()
    for i in range(len(query)):  # input-target index
        for j in range(len(query[0])):  # spatial index
            x_j = x[j]
            if i == 0:
                role = 2  # query input
                tok = (x_j, query[i][j], role)
                input_tokens.append(tok)
            else:
                role = 3  # query target
                tok = (x_j, query[i][j], role)
                target_tokens.append(tok)
    input_tokens = jnp.array(input_tokens)
    target_tokens = jnp.array(target_tokens)
    return input_tokens, target_tokens


def batch_task(
    x: jax.Array, examples: jax.Array, key: jax.Array, *, n: int, k: int
) -> tuple[jax.Array, jax.Array]:
    """
    create batch of sampled tasks

    parameters
    ----------
    x: jax.Array
        spatial coordinates
    examples: jax.Array
        input-target examples
    key: jax.Array
        random number generator
    n: int
        number of sampled tasks
    k: int
        number of context examples

    returns
    -------
    input_batch: jax.Array
        sampled input tasks
    target_batch: jax.Array
        sampled target tasks
    """
    input_batch = list()
    target_batch = list()
    for i in range(n):
        key, subkey = jr.split(key)
        context, query = sample_task(examples, subkey, k=k)
        input_tokens, target_tokens = tokenize(x, context, query)
        input_batch.append(input_tokens)
        target_batch.append(target_tokens)
    input_batch = jnp.array(input_batch)
    target_batch = jnp.array(target_batch)
    return input_batch, target_batch

def main():
    pass

if __name__ == "__main__":
    main()