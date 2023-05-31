import numpy as np

def center_kernel(kernel):
    height, width = kernel.shape
    x, y, w, h = bounding_box(kernel)
    # Create new blank image and shift ROI to new coordinates
    ROI = kernel[y:y + h, x:x + w]
    # find baricenter of kernel
    baricenter_x = 0
    baricenter_y = 0
    for i, column in enumerate(ROI):
        for j, pixel in enumerate(column):
            baricenter_x += i*pixel
            baricenter_y += j*pixel
    baricenter_x = baricenter_x / kernel.sum()
    baricenter_y = baricenter_y / kernel.sum()
    baricenter_x = int(np.rint(baricenter_x))
    baricenter_y = int(np.rint(baricenter_y))
    x = width // 2 - baricenter_x
    y = height // 2 - baricenter_y
    mask = np.pad(ROI, [(x,x+1),(y,y+1)])
    # import matplotlib.pyplot as plt
    # plt.imshow(kernel, cmap="gray")
    # plt.show()
    # plt.imshow(mask[:height,:width], cmap="gray")
    # plt.show()
    return mask[:height,:width]

def bounding_box(kernel):
    # finds the box that contains the kernel
    # outputs the coordinates of the upper left point
    # and the width and height of the box
    col_sum = np.sum(kernel, axis=0)
    x = 0
    for i, p in enumerate(col_sum):
        if p == 0:
            x = i
        else:
            break
    x_rev = 0
    for i, p in enumerate(reversed(col_sum)):
        if p == 0:
            x_rev = i
        else:
            break
    w = col_sum.size - x - x_rev
    row_sum = np.sum(kernel, axis=1)
    y = 0
    for i, p in enumerate(row_sum):
        if p == 0:
            y = i
        else:
            break
    y_rev = 0
    for i, p in enumerate(reversed(row_sum)):
        if p == 0:
            y_rev = i
        else:
            break
    h = col_sum.size - y - y_rev
    return x, y, w, h
