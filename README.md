# GPU Queue Testing

## How to Run
1. Edit `kernels/queue_dispatch.cl` : uncomment queue file you want to test (MS, SFQ, or TZ)
2. `mkdir build && cd build && cmake .. && make`
3. `./queue_test`
