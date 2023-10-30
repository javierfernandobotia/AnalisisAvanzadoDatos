def Main_train_DEC(x, model, target_distribution, past_labels, index_array, maxiter, batch_size, index, loss, update_interval, tol):
    import numpy as np
    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            q, _  = model.predict(x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p
            new_labels = q.argmax(1)

        # check stop criterion
            delta_label = np.sum(new_labels != past_labels).astype(np.float32) / new_labels.shape[0]
            past_labels = np.copy(new_labels)
            if (ite > 0) and (delta_label < tol):
                print('delta_label ', delta_label, '< tol ', tol)
                print('Tolerance error is reached')
                break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0
    return index, loss