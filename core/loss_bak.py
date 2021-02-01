    c = 0

    for x in x_real:
        # save real(1)
        img = transforms.ToPILImage()(x)
        img.save('real'+str(c)+'.jpg')
        x_real_tensors.append(tensor(img))

    print('x_real_tensors:', len(x_real_tensors))

    for x in x_fake:
        # save fake(1)
        img = transforms.ToPILImage()(x)
        img.save('fake'+str(c)+'.jpg')
        c = c + 1
        x_fake_tensors.append(tensor(img))
    
    # stack real
    stacked_tensor = torch.stack(x_real_tensors).to('cpu')
    print("stacked_tensor:", stacked_tensor.shape)

    # stack fake
    stacked_fake_tensor = torch.stack(x_fake_tensors).to('cpu')
    print("stacked_fake_tensor:", stacked_fake_tensor.shape)

    detected = mtcnn.detect(stacked_tensor)

    if mtcnn.detect(stacked_tensor)[1].dtype is np.dtype('float32'):
        selected_real = True
        real_aligned, prob = mtcnn(stacked_tensor, return_prob=True)
        stacked_real_aligned = torch.stack(real_aligned)
        print("stacked_real_aligned:", stacked_real_aligned.shape)

        real_embeddings = resnet(stacked_real_aligned).detach().cpu()
        print('real_embeddings:', real_embeddings.shape)

    else:
        selected_real = False
    
    c = 0

    if mtcnn.detect(stacked_fake_tensor)[1].dtype is np.dtype('float32'):
        selected_fake = True
        fake_aligned, prob = mtcnn(stacked_fake_tensor, return_prob=True)
        stacked_fake_aligned = torch.stack(fake_aligned)
        print("stacked_fake_aligned:", stacked_fake_aligned.shape)

        fake_embeddings = resnet(stacked_fake_aligned).detach().cpu()
        print('fake_embeddings:', fake_embeddings.shape)
    else:
        selected_fake = False

    # minus
    print("selected_real:", selected_real)
    print("selected_fake:", selected_fake)

    if selected_real and selected_fake:
        print("match_loss:", torch.linalg.norm(real_embeddings - fake_embeddings, 1, -1).mean())
    else:
        print("match_loss:", 0)
