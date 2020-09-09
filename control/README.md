# Usage
```python
from serial import Serial
from control import Motors

m = Motors(Serial('/dev/ttyS3', 115200, timeout=1))
while m.read() < 100:
    m.go(90, 1, 20)
m.stop()
```