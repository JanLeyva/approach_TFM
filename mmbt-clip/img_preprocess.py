def slice_image(im, desired_size):
    '''
    Resize and slice image
    '''
    old_size = im.size
    ratio = float(desired_size)/min(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)    
    ar = np.array(im)
    images = []
    if ar.shape[0] < ar.shape[1]:
        middle = ar.shape[1] // 2
        half = desired_size // 2
        
        images.append(Image.fromarray(ar[:, :desired_size]))
        images.append(Image.fromarray(ar[:, middle-half:middle+half]))
        images.append(Image.fromarray(ar[:, ar.shape[1]-desired_size:ar.shape[1]]))
    else:
        middle = ar.shape[0] // 2
        half = desired_size // 2
        
        images.append(Image.fromarray(ar[:desired_size, :]))
        images.append(Image.fromarray(ar[middle-half:middle+half, :]))
        images.append(Image.fromarray(ar[ar.shape[0]-desired_size:ar.shape[0], :]))

    return images
  
 def resize_pad_image(im, desired_size):
    '''
    Resize and pad image to a desired size
    '''
    old_size = im.size
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)

    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im