import numpy as np
import matplotlib.pyplot as plt

"""
Relative risk of maternal death between pregnant and nonpregnant women with coronavirus disease.
Data from 2019, Mexico
"""

RR = np.array([0.83,2.24,1.66,1.49,1.50,2.30])

ci_upper = np.array([3.41,3.90,2.71,2.27,2.36,3.85])

ci_lower = np.array([0.20,1.29,1.02,0.97,0.95,1.37])

x = np.array([0,1,2,3,4,5])


fig, ax = plt.subplots()
ax.plot(x,RR,"o-", label="Observed data")
ax.set_xticks(x)
ax.set_xticklabels(["15-19","20-24","25-29","30-34","35-39","40-44"])
ax.fill_between(x, ci_upper, ci_lower, color='b', alpha=.1, label="95% Confidence interval")
ax.legend(loc="lower right",prop={'size': 7})
plt.ylabel("Risk ratio")
plt.title("Risk ratio of maternal death between mexican\n pregnant and non-pregnant women with Sars-CoV-2")
plt.show()
