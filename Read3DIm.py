# allowable extensions for image files
import os
import io
import numpy as np

allowable_extensions = ['.png', '.tif', '.tiff', '.bmp', '.dib', '.jpg', '.jpeg']


def get_index_from_file_name(path_name):
    ''' return index from path_name

        e.g. for image0001 index is 1

     '''

    file_name = path_splitted = os.path.split(path_name)[-1]

    file_name = os.path.splitext(file_name)[0]

    i = len(file_name) - 1

    while (file_name[i].isdigit() == True and i >= 0):
        i -= 1

    if i == len(file_name) - 1:  # there are no any digits

        return -1

    return int(file_name[i + 1:])


def read_slices(path_to_first_slice):
    # check correctness of type of input parameter

    if not isinstance(path_to_first_slice, str):
        raise TypeError('Parameter path_to_first_slice has to be string!')

    path_splitted = os.path.split(path_to_first_slice)

    first_slice_file_name = path_splitted[-1]

    filename_splitted = os.path.splitext(first_slice_file_name)

    first_slice_extension = filename_splitted[-1]

    if not first_slice_extension in allowable_extensions:
        raise ValueError(first_slice_file_name + ' is unsupported file format!')

    # open first slice

    try:

        slice1 = io.imread(path_to_first_slice)

    except FileNotFoundError:

        raise ValueError('File not found for : ' + path_to_first_slice)

    except PermissionError:

        raise ValueError('Permission error for : ' + path_to_first_slice)

    except:

        raise ValueError('Reading error for : ' + path_to_first_slice)

    # check dimensions and bit depth

    if slice1.ndim != 2:
        raise ValueError(path_to_first_slice + ' must be grayscale 2D image')

    if slice1.dtype != np.uint8:  # and slice1.dtype != np.uint16:

        raise ValueError(path_to_first_slice + ' has wrong bit depth.')

    # forming of template for file names of slices

    template_name = filename_splitted[0]

    # trim digits on end of filename and calculate number of digits

    digits_len = 0

    while (len(template_name) > 0 and template_name[-1].isdigit() == True):
        template_name = template_name[:-1]

        digits_len += 1


# checking of digits in the end of file name

        if digits_len < 3:
            raise ValueError('Incorrect name of first slice, 3 or more digits should be in the end of file name: ' +

                             first_slice_file_name)

        search_template = '%s' % template_name  # string copying via formatting

        for i in range(0, digits_len):
            search_template += "?"

        # looking for maximal index of slice

        files_list = glob.glob(os.path.join(path_splitted[0], search_template + first_slice_extension))

        max_slice_index = 0

        for file_name in files_list:

            slice_index = get_index_from_file_name(file_name)

            if slice_index > max_slice_index:
                max_slice_index = slice_index

        first_slice_index = int(filename_splitted[0][-digits_len:])

        template_name = os.path.join(path_splitted[0], template_name + "{0:0" + str(digits_len) + "}" +

                                     first_slice_extension)

        # memory allocation for image

        i3d = np.zeros((max_slice_index - first_slice_index + 1,) + slice1.shape, dtype=slice1.dtype)

        i3d[0] = slice1

        # loop through all indexes

        k = 1

        for i in range(first_slice_index + 1, max_slice_index + 1):
            i3d[k] = io.imread(template_name.format(i))

            k += 1

        return i3d

