import cv2 as cv
from numpy import shape,zeros,max,min

def stretch(matrix):
    print('Matrix = %d x %d'%matrix.shape)
    input_min = matrix.min()
    input_max = matrix.max()
    print(input_min,input_max)
    output_max = 255
    output_min = 0
    if input_max == output_max and input_min == output_min:
        return matrix
    return_matrix = zeros(matrix.shape,dtype='uint8')
    for i in range(0,matrix.shape[0]):
        for j in range(0,matrix.shape[1]):
            return_matrix[i][j] = int((matrix[i][j]-input_min)*(output_max-output_min)/(input_max-input_min) + output_min)
    print(return_matrix.min(),return_matrix.max())
    return  return_matrix


def contrast_stretch(filepath):
    img = cv.imread(filepath)
    print(img.shape)
    return_img = zeros(img.shape)

    red = img[:,:,0]
    blue = img[:,:,1]
    green = img[:,:,2]

    return_img[:,:,0] = stretch(red)
    # print(red)
    return_img[:,:,1] = stretch(blue)
    # print(blue)
    return_img[:,:,2] = stretch(green)
    # print(green)
    print(return_img.shape)
    return return_img

if __name__ == '__main__':
    filepath = 'train/Arnab.jpg'
    # cv.imshow('Arnab',contrast_stretch(filepath))
    # cv.waitKey(0)
    cv.imwrite('train/mod_Arnab.jpg',contrast_stretch(filepath))