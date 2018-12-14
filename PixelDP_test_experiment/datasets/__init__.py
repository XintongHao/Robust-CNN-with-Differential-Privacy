def build_input(dataset, data_path, batch_size, standardize_images, mode):
    if dataset == 'mnist':
        from datasets import mnist
        return mnist.build_input(data_path, batch_size, standardize_images, mode)

    else:
        raise ValueError("Dataset {} not supported".format(dataset))
