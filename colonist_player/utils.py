import msgpack
import base64

def parse_websocket_message(base64_message):
    base64_bytes = base64_message.encode("utf-8")
    message_bytes = base64.b64decode(base64_bytes)
    message = msgpack.unpackb(message_bytes, raw=True)
    return bytes_dict_to_json(message)


def bytes_dict_to_json(data):
    """
    Transforms {b"id": b"9", b"data": {b"currentTurnState": 0}}
    into {"id": "9", "data": {"currentTurnState": 0}}
    """
    if isinstance(data, bytes):
        return data.decode("utf-8")
    elif isinstance(data, dict):
        return {
            k.decode("utf-8"): bytes_dict_to_json(v) for k, v in data.items()
        }
    elif isinstance(data, list):
        return [bytes_dict_to_json(v) for v in data]
    else:
        return data

print(parse_websocket_message("gqJpZKIxMKRkYXRhpHRCQUI="))
# Example
# base64_message = "gqJpZKE5pGRhdGGGsGN1cnJlbnRUdXJuU3RhdGUAsmN1cnJlbnRBY3Rpb25TdGF0ZQC2Y3VycmVudFR1cm5QbGF5ZXJDb2xvcgGubGFzdFN0YXRlU3RhcnTPAAABfi/nXHmydGltZVRvV2FpdEZvclN0YXRlKKtjdXJyZW50VGltZc8AAAF+L+dceQ=="
# data = parse_websocket_message(base64_message)
# print(data)