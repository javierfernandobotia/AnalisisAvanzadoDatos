def Initial_train_DEC(x, model, index_array, update_interval, maxiter, batch_size, index, loss):
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    return index, loss