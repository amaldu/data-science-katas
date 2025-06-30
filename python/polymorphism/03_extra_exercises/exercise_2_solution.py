



###----------------------- Exercise 2 ----------------------###

# strategy pattern with polymorphism
# import statistics
# Implement a strategy pattern where a DataAnalyzer class can work with different analysis strategies like
#  MeanStrategy, MedianStrategy, and ModeStrategy. Each strategy must have a method analyze(data: list) that implements
# the mean, the median and the mode from the statistics library

import statistics


class Strategy:
    def analyze(self, data):
        raise NotImplementedError

class MeanStrategy(Strategy):
    def analyze(self,data):
        return statistics.mean(data)

class MedianStrategy(Strategy):
    def analyze(self,data):
        return statistics.median(data)

class ModeStrategy(Strategy):
    def analyze(self,data):
        return statistics.mode(data)
    
class DataAnalyzer():
    def __init__(self, strategy:Strategy):
        self.strategy = strategy

    def run(self, data):
        return self.strategy.analyze(data) 


data = [1, 2, 2, 3, 4]

mean_analyzer = DataAnalyzer(MeanStrategy())
median_analyzer = DataAnalyzer(MedianStrategy())
mode_analyzer = DataAnalyzer(ModeStrategy())

print("Mean:", mean_analyzer.run(data))
print("Median:", median_analyzer.run(data))
print("Mode:", mode_analyzer.run(data))


"""
Mean: 2.4
Median: 2
Mode: 2
"""