import onnx
import onnxruntime as rt
from io import BytesIO
from .device import ORTDeviceInfo


def InferenceSession_with_device(onnx_model_or_path, device_info : ORTDeviceInfo):
    """
    Construct onnxruntime.InferenceSession with this Device.

     device_info     ORTDeviceInfo

    can raise Exception
    """

    if isinstance(onnx_model_or_path, onnx.ModelProto):
        b = BytesIO()
        onnx.save(onnx_model_or_path, b)
        onnx_model_or_path = b.getvalue()

    device_ep = device_info.get_execution_provider()
    if device_ep not in rt.get_available_providers():
        raise Exception(f'{device_ep} is not avaiable in onnxruntime')

    ep_flags = {}
    if device_ep in ['CUDAExecutionProvider','DmlExecutionProvider']:
        ep_flags['device_id'] = device_info.get_index()

    sess_options = rt.SessionOptions()
    # On a GPU EP, surface warnings so a silent CPU fallback (e.g. a CUDA/cuDNN
    # mismatch) shows up in the logs instead of hiding behind a 'GPU' status.
    sess_options.log_severity_level = 2 if device_ep in ('CUDAExecutionProvider', 'DmlExecutionProvider') else 4
    sess_options.log_verbosity_level = -1
    if device_ep == 'DmlExecutionProvider':
        sess_options.enable_mem_pattern = False
    sess = rt.InferenceSession(onnx_model_or_path, providers=[ (device_ep, ep_flags) ], sess_options=sess_options)
    return sess
