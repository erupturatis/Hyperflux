import random
import numpy as np

scale = 3
initSum = 370
percentage = 0.5
MAX = 35
step = 1.5

def foolproof(c,S):
    return (c * S) / (10 * (MAX-c))

def get_discounted_sum(initSum, positions, currentPrice):
    newSum = initSum
    for position in positions:
        x_price_increase = (MAX / position["price"] - 1) * 10
        newSum -= x_price_increase * position["sum"]
    return newSum

def get_positions_profit(endPrice):
    profits = 0
    for position in positions:
        x_price_increase = (endPrice / position["price"] - 1) * 10
        profits -= x_price_increase * position["sum"]

    return profits

expected_return = 0
trials = 500

# for experiment in range(trials):
#
#     scale = 2
#     initialize = 370
#     initSum = 370
#     percentage = 0.5
#     step = 1
#     for itr in range(20):
#         sample = np.random.exponential(scale) + 18
#         peak_vix = int(min(sample, 33))
#         end_vix = random.randint(15, 16)
#         positions = []
#
#         for i in range(18,peak_vix,step):
#             pos = foolproof(i, get_discounted_sum(initSum, positions, i))
#             # print("Current price", i, " Opened position:", pos * percentage, "Current sum", get_discounted_sum(initSum, positions, i))
#             positions.append({
#                 'price': i,
#                 'sum': pos * percentage,
#             })
#
#         pr = get_positions_profit(end_vix)
#         # print("Final price", end_vix, "Final profit", pr)
#         initSum += pr
#         # print(initSum)
#
#     expected_return += initSum / initialize
#
# print("EXPECTED RETURNS")
# print(expected_return / trials)

positions = [{
    "price": 20.3,
    "sum": 30,
}]
currentVixFuture = 22.21
pos = foolproof(currentVixFuture, get_discounted_sum(initSum, positions, currentVixFuture))
sumToBeInvested = pos * 0.5
print(sumToBeInvested)