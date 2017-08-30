from progressbar import ProgressBar
import time

p = ProgressBar(maxval=50).start()  # 最大値100
for i in range(50):
    p.update(i+1)
    time.sleep(0.01)
p.finish()