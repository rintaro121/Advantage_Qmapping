import os
import time


MAX_CNT = 100
cnt = 0


one = """
............................
............................
............................
............................
............................
...........#####............
..........######............
.........#######............
........###.####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............####............
............................
............................
............................
"""

two = """
............................
............................
............................
............................
............#######.........
.........#############......
.......################.....
......#################.....
.....#######.....######.....
....#######......######.....
...............#######......
..............########......
.............########.......
...........#########........
.........#########..........
........########............
.......#######..............
......#######...............
.....#######................
....########................
....#####################...
....#####################...
.....###################....
............................
............................
............................
............................
............................
"""

three = """
............................
............................
............................
............................
......###############.......
......###############.......
......###############.......
................#####.......
...............#####........
.............######.........
............######..........
..........#######...........
........########............
........##########..........
.........##########.........
...............#####........
................####........
................####........
.....##........#####........
.....###......#####.........
.....####....######.........
.....#############..........
......###########...........
.......#########............
............................
............................
............................
............................
"""

four = """
............................
............................
............................
............................
.................#..........
................##..........
...............###..........
..............####..........
.............#####..........
............######..........
...........###.###..........
..........###..###..........
.........###...###..........
........###....###..........
.......###.....###..........
......###......###..........
.....###.......###..........
....######################..
...#######################..
...............###..........
...............###..........
...............###..........
...............###..........
...............###..........
............................
............................
............................
............................


"""

five = """
............................
............................
............................
............................
.....###################....
.....###################....
.....###....................
.....###....................
.....###....................
.....###....................
.....###...######...........
.....###..########..........
.....##############.........
.....#####......####........
.....####........####.......
.................####.......
.................####.......
.................####.......
.................####.......
.................###........
.....##.........####........
.....###.......####.........
.....####.....####..........
......###########...........
.......#########............
............................
............................
............................
"""


if __name__ == "__main__":

    disp_num = [one, two, three, four, five]

    def count_down():
        for num in disp_num:
            print(num)
            time.sleep(1)
            os.system('clear')
            


    count_down()
    print('End')