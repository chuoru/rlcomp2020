# coding: utf-8
"""

Abstract::
    - 
History::
    - Ver.      Date            Author        History
    - 
Copyright (C) 2020 HACHIX Corporation. All Rights Reserved.
"""
# 標準ライブラリ


# 関連外部ライブラリ


# 内部ライブラリ
MAP = [[550,-1,-1,-2,-2,-2,-2,150,0,0,0,-3,-2,400,-3,650,-2,-1,0,0,0],[350,-1,-1,-2,0,0,-2,-2,-2,-1,0,-3,-3,-2,300,-2,-1,650,-2,-1,0],[-1,-1,450,-2,0,0,0,0,0,0,0,0,0,-3,-3,-3,-1,-1,-3,0,0],[-1,-1,-2,-2,0,-3,-2,0,0,-3,-3,-3,0,0,0,0,0,0,-3,0,150],[0,0,-2,0,-3,-3,-3,0,-1,-2,400,-3,-3,0,-2,0,-1,0,-3,0,0],[0,200,-2,0,-3,250,-1,0,0,-1,-2,-3,-2,0,-1,300,-1,0,-3,-1,0],[-3,-3,-2,0,-3,-3,-3,0,0,0,0,-3,-3,-3,-2,-2,-2,0,-2,-2,0],[-1,-3,-2,0,0,-2,0,0,-3,-3,-3,-3,150,-3,0,0,0,0,-2,200,0],[800,-3,-2,-3,450,-2,0,-3,-3,200,-1,250,-1,-3,0,-1,-1,0,0,0,0]]
maxx = len(MAP[0])
maxy = len(MAP)
print(' '*2, end = '|')
for i in range(maxx):
    print(f'{i:^3}', end = '|')
print()
index = 0
for element in MAP:
    t = map(str, element)
    print(f'{index:<2}', end = '|')

    for word in t:
        print(f'{word:^3}', end=' ')
    print()
    index += 1
