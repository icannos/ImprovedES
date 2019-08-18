from dinomodels import smallDino

import numpy as np


specimen = smallDino.init_parameters()
specimen1 = smallDino.init_parameters()

lw = []

for w in specimen:
    lw.append(np.random.normal(w.astype(np.float64), 0.6))


print(smallDino.distance(specimen, lw))
