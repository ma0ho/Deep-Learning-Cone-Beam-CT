import tensorflow as tf
import re

def read( filename, data_type = tf.float32 ):
    assert( data_type == tf.float32 or data_type == tf.float64 )

    file_handle = open( filename, 'r' )
    file_contents = file_handle.read()
    file_handle.close()

    data = []
    regex = re.compile( r"@\s\d*\n[\d.]+\s+[\d.]+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+\n([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+([\d\-.E]+)\s+" )
    for match in regex.finditer( file_contents ):
        data += [ float( x ) for x in match.groups() ]

    assert( len(data) % 12 == 0 )

    result = tf.constant( data, dtype = data_type )

    return tf.reshape( result, [int( len(data)/12 ), 3, 4] )


