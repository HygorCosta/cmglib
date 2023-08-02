import numpy as np
from tqdm import tqdm
from time import sleep

def parse_all_string(all_string, dtype=float, shape=None):
    """Convert string in *ALL format to a numpy array.

    Arguments:
        all_string {str} -- String object containing *ALL lines.

    Keyword Arguments:
        dtype {numpy.{datatype}} -- numpy datatype selected for output (default: {np.float})
        shape {tuple} -- shape of the output array (default: {None})
    """

    all_string = all_string.strip().replace('\n', ' ').split(' ')

    all_values = []
    for i in tqdm(all_string, miniters=int(len(all_string)/100)):
        if '*' not in i:
            all_values.append(dtype(i))
        else:
            mult, value = i.split('*')
            all_values.extend(mult * [dtype(value)])

    all_array = np.array(all_values)

    if shape is not None:
        return all_array.reshape(shape)
    else:
        return all_array
