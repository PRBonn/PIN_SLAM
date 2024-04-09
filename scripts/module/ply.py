# Copyright: jedeschaud
# Basic libs
import numpy as np
import sys

# Define PLY types
ply_dtypes = dict([
  (b'char', 'i1'),
  (b'int8', 'i1'),
  (b'uchar', 'b1'),
  (b'uchar', 'u1'),
  (b'uint8', 'u1'),
  (b'short', 'i2'),
  (b'int16', 'i2'),
  (b'ushort', 'u2'),
  (b'uint16', 'u2'),
  (b'int', 'i4'),
  (b'int32', 'i4'),
  (b'uint', 'u4'),
  (b'uint32', 'u4'),
  (b'float', 'f4'),
  (b'float32', 'f4'),
  (b'double', 'f8'),
  (b'float64', 'f8')
])


# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}


def parse_header(plyfile, ext):
  # Variables
  line = []
  properties = []
  num_points = None

  while b'end_header' not in line and line != b'':
    line = plyfile.readline()
  
    if b'element' in line:
      line = line.split()
      num_points = int(line[2])

    elif b'property' in line:
      line = line.split()
      properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

  return num_points, properties


def read_ply(filename):
  """
  Read ".ply" files

  Parameters
  ----------
  filename : string
    the name of the file to read.

  Returns
  -------
  result : array
    data stored in the file

  Examples
  --------
  Store data in file

  >>> points = np.random.rand(5, 3)
  >>> values = np.random.randint(2, size=10)
  >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

  Read the file

  >>> data = read_ply('example.ply')
  >>> values = data['values']
  array([0, 0, 1, 1, 0])
  
  >>> points = np.vstack((data['x'], data['y'], data['z'])).T
  array([[ 0.466  0.595  0.324]
       [ 0.538  0.407  0.654]
       [ 0.850  0.018  0.988]
       [ 0.395  0.394  0.363]
       [ 0.873  0.996  0.092]])

  """

  with open(filename, 'rb') as plyfile:
    # Check if the file start with ply
    if b'ply' not in plyfile.readline():
      raise ValueError('The file does not start whith the word ply')

    # get binary_little/big or ascii
    fmt = plyfile.readline().split()[1].decode()
    if fmt == "ascii":
      raise ValueError('The file is not binary')

    # get extension for building the numpy dtypes
    ext = valid_formats[fmt]

    # Parse header
    num_points, properties = parse_header(plyfile, ext)

    # Get data
    data = np.fromfile(plyfile, dtype=properties, count=num_points)

  return data


def header_properties(field_list, field_names):
  # List of lines to write
  lines = []

  # First line describing element vertex
  lines.append('element vertex %d' % field_list[0].shape[0])

  # Properties lines
  i = 0
  for fields in field_list:
    for field in fields.T:
      lines.append('property %s %s' % (field.dtype.name, field_names[i]))
      i += 1

  return lines


def write_ply(filename, field_list, field_names):
  """
  Write ".ply" files

  Parameters
  ----------
  filename : string
    the name of the file to which the data is saved. A '.ply' extension will be appended to the 
    file name if it does no already have one.

  field_list : list, tuple, numpy array
    the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
    tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
    as one field. 

  field_names : list
    the name of each fields as a list of strings. Has to be the same length as the number of 
    fields.

  Examples
  --------
  >>> points = np.random.rand(10, 3)
  >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

  >>> values = np.random.randint(2, size=10)
  >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

  >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
  >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
  >>> write_ply('example3.ply', [points, colors, values], field_names)

  """

  # Format list input to the right form
  field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
  for i, field in enumerate(field_list):
    if field is None:
      print('WRITE_PLY ERROR: a field is None')
      return False
    elif field.ndim > 2:
      print('WRITE_PLY ERROR: a field have more than 2 dimensions')
      return False
    elif field.ndim < 2:
      field_list[i] = field.reshape(-1, 1)

  # check all fields have the same number of data
  n_points = [field.shape[0] for field in field_list]
  if not np.all(np.equal(n_points, n_points[0])):
    print('wrong field dimensions')
    return False    

  # Check if field_names and field_list have same nb of column
  n_fields = np.sum([field.shape[1] for field in field_list])
  if (n_fields != len(field_names)):
    print(n_fields, 'fields instead of', len(field_names))
    return False

  # Add extension if not there
  if not filename.endswith('.ply'):
    filename += '.ply'

  # open in text mode to write the header
  with open(filename, 'w') as plyfile:

    # First magical word
    header = ['ply']

    # Encoding format
    header.append('format binary_' + sys.byteorder + '_endian 1.0')

    # Points properties description
    header.extend(header_properties(field_list, field_names))

    # End of header
    header.append('end_header')

    # Write all lines
    for line in header:
      plyfile.write("%s\n" % line)


  # open in binary/append to use tofile
  with open(filename, 'ab') as plyfile:

    # Create a structured array
    i = 0
    type_list = []

    for fields in field_list:
      for field in fields.T:
        type_list += [(field_names[i], field.dtype.str)]
        i += 1
    data = np.empty(field_list[0].shape[0], dtype=type_list)
    i = 0
    for fields in field_list:
      for field in fields.T:
        data[field_names[i]] = field
        i += 1

    data.tofile(plyfile)

  return True