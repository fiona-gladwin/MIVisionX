


# RocalStatus
from rali_pybind.types import OK
from rali_pybind.types import CONTEXT_INVALID
from rali_pybind.types import RUNTIME_ERROR
from rali_pybind.types import UPDATE_PARAMETER_FAILED
from rali_pybind.types import INVALID_PARAMETER_TYPE

#  RocalProcessMode
from rali_pybind.types import GPU
from rali_pybind.types import CPU

#  RocalTensorOutputType
from rali_pybind.types import FLOAT
from rali_pybind.types import FLOAT16
from rali_pybind.types import UINT8

# RocalImageSizeEvaluationPolicy
from rali_pybind.types import MAX_SIZE
from rali_pybind.types import USER_GIVEN_SIZE
from rali_pybind.types import MOST_FREQUENT_SIZE
from rali_pybind.types import MAX_SIZE_ORIG
from rali_pybind.types import USER_GIVEN_SIZE_ORIG


#      RocalImageColor
from rali_pybind.types import RGB
from rali_pybind.types import BGR
from rali_pybind.types import GRAY
from rali_pybind.types import RGB_PLANAR

#     RocalTensorLayout
from rali_pybind.types import NHWC
from rali_pybind.types import NCHW

#     RocalDecodeDevice
from rali_pybind.types import HARDWARE_DECODE
from rali_pybind.types import SOFTWARE_DECODE

#     RocalDecodeDevice
from rali_pybind.types import DECODER_TJPEG
from rali_pybind.types import DECODER_OPENCV
from rali_pybind.types import DECODER_HW_JEPG
from rali_pybind.types import DECODER_VIDEO_FFMPEG_SW
from rali_pybind.types import DECODER_VIDEO_FFMPEG_HW


_known_types={


	OK : ("OK", OK),
    CONTEXT_INVALID : ("CONTEXT_INVALID", CONTEXT_INVALID),
	RUNTIME_ERROR : ("RUNTIME_ERROR", RUNTIME_ERROR),
    UPDATE_PARAMETER_FAILED : ("UPDATE_PARAMETER_FAILED", UPDATE_PARAMETER_FAILED),
	INVALID_PARAMETER_TYPE : ("INVALID_PARAMETER_TYPE", INVALID_PARAMETER_TYPE),

	GPU : ("GPU", GPU),
    CPU : ("CPU", CPU),
	FLOAT : ("FLOAT", FLOAT),
    FLOAT16 : ("FLOAT16", FLOAT16),
    UINT8 : ("UINT8", UINT8),


	MAX_SIZE : ("MAX_SIZE", MAX_SIZE),
    USER_GIVEN_SIZE : ("USER_GIVEN_SIZE", USER_GIVEN_SIZE),
	MOST_FREQUENT_SIZE : ("MOST_FREQUENT_SIZE", MOST_FREQUENT_SIZE),
    MAX_SIZE_ORIG : ("MAX_SIZE_ORIG", MAX_SIZE_ORIG),
    USER_GIVEN_SIZE_ORIG : ("USER_GIVEN_SIZE_ORIG", USER_GIVEN_SIZE_ORIG),

	NHWC : ("NHWC", NHWC),
    NCHW : ("NCHW", NCHW),
	BGR : ("BGR", BGR),
    RGB : ("RGB", RGB),
	GRAY : ("GRAY", GRAY),
    RGB_PLANAR : ("RGB_PLANAR", RGB_PLANAR),

    HARDWARE_DECODE : ("HARDWARE_DECODE", HARDWARE_DECODE),
    SOFTWARE_DECODE : ("SOFTWARE_DECODE", SOFTWARE_DECODE),
    
    DECODER_TJPEG : ("DECODER_TJPEG", DECODER_TJPEG),
    DECODER_OPENCV : ("DECODER_OPENCV", DECODER_OPENCV),
    DECODER_HW_JEPG : ("DECODER_HW_JEPG", DECODER_HW_JEPG),
    DECODER_VIDEO_FFMPEG_SW : ("DECODER_VIDEO_FFMPEG_SW", DECODER_VIDEO_FFMPEG_SW),
    DECODER_VIDEO_FFMPEG_HW : ("DECODER_VIDEO_FFMPEG_HW", DECODER_VIDEO_FFMPEG_HW)
}


def data_type_function(dtype):
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        return ret
    else:
        raise RuntimeError(str(dtype) + " does not correspond to a known type.")
