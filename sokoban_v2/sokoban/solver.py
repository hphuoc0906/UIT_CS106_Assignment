import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
import math
global posWalls, posGoals
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

"""Load puzzles and define the rules of sokoban"""

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     

    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]] 
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])
    return temp

def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = collections.deque([[0]])
    temp = []
    ### CODING FROM HERE ###

def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    start =  time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    temp = []
    ### CODING FROM HERE ###
    while frontier.isEmpty() == False: # trong khi hàng đợi ưu tiên frontier
        node = frontier.pop() # lấy nút trạng thái có chi phí thấp nhất trong hàng đợi
        node_action = actions.pop() # lấy hành động của nút mới ra hàng chờ
        if isEndState(node[-1][-1]): # nếu là trạng thái kết thúc
            temp += node_action[1:] # lấy chuỗi kết quả
            break # tạm dừng xét
        if node[-1] not in exploredSet: # nếu trạng thái hiện tại chưa tồn tại trong tập đã xét
            exploredSet.add(node[-1]) # thêm vào tập đã xét
            for action in legalActions(node[-1][0], node[-1][1]): # xét các hành động kế tiếp cho phép trong trạng thái hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # lấy ra trạng thái mới theo hành động tương ứng
                if isFailed(newPosBox): # nếu hành động có khả năng thất bại
                    continue # thì bỏ qua
                new_action = node_action + [action[-1]] # kết hợp với các hành động trước đó
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(new_action[1:])) # thêm trạng thái mới vào trong hàng đợi
                actions.push(new_action, cost(new_action[1:])) # thêm hành động của trạng thái mới vào trong hàng chờ hành động
    end =  time.time()
    print(end - start)
    print(len(exploredSet))
    return temp # trả về kết quả

def heuristic(posPlayer, posBox):
    """A heuristic function to calculate the overall distance between the else boxes and the else goals"""
    distance = 0 # khai báo 1 biến trả về khoảng cách
    completes = set(posGoals) & set(posBox) # lấy ra những hộp đã được xếp đúng vị trí
    # set trong python được sắp xếp theo thứ tự nhập vào
    sortposBox = list(set(posBox).difference(completes)) # lấy ra vị trí những hộp chưa vào vị trí đích
    sortposGoals = list(set(posGoals).difference(completes)) # lấy những vị trí đích đến chưa có hộp nào chồng lên
    for i in range(len(sortposBox)): # với mỗi vị trí hộp
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1])) # lấy khoảng cách của vi trí đích và hộp tương ứng theo thứ tự nhập vào
    return distance # trả về khoảng cách

def customHeuristic1(posPlayer, posBox): # Hàm Heuristic tự thiết kế thay bằng khoảng cách Manhattan
    distance = 0 # khai báo 1 biến trả về khoảng cách
    completes = set(posGoals) & set(posBox) # lấy ra những hộp đã được xếp đúng vị trí
    sortposBox = sorted(list(set(posBox).difference(completes)), key=lambda x: (x[0], x[1])) # sắp xếp các hộp
    sortposGoals = sorted(list(set(posGoals).difference(completes)), key=lambda x: (x[0], x[1])) # sắp xếp các vị trí đích
    
    for i in range(len(sortposBox)): # tính toán tổng khoảng cách Manhattan giữa 2 điểm tương ứng được đưa vào 
        distance += (abs(sortposBox[i][0] - sortposGoals[i][0])) + (abs(sortposBox[i][1] - sortposGoals[i][1]))
    return distance # trả về khoảng cách


def euclidDistance(pos1, pos2): # tính khoảng cách euclid giữa 2 điểm
    x1, y1 = pos1 # lấy tọa độ điểm 1
    x2, y2 = pos2 # lấy tọa độ điểm 2
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) # tính khoảng cách euclid

def customHeuristic2(posPlayer, posBox): # Hàm Heuristic tự thiết kế thay bằng khoảng cách euclid   
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = list(set(posBox).difference(completes)) # lấy ra vị trí những hộp chưa vào vị trí đích
    sortposGoals = list(set(posGoals).difference(completes)) # lấy những vị trí đích đến chưa có hộp nào chồng lên
    for i in range(len(sortposBox)): # lấy tổng khoảng cách Euclid
        distance += euclidDistance(sortposBox[i], sortposGoals[i])
    return distance

def customHeuristic3(posPlayer, posBox): # Hàm Heuristic tự thiết kế thay bằng khoảng cách euclid nhưng các điểm đã sắp xếp theo x, y tăng dần
    distance = 0
    completes = set(posGoals) & set(posBox)
    sortposBox = sorted(list(set(posBox).difference(completes)), key=lambda x: (x[0], x[1])) # sắp xếp các hộp
    sortposGoals = sorted(list(set(posGoals).difference(completes)), key=lambda x: (x[0], x[1])) # sắp xếp các vị trí đích
    for i in range(len(sortposBox)):
         distance += euclidDistance(sortposBox[i], sortposGoals[i])
    return distance

class MinimumCostMaxFlow: # Khai báo bài toán luồng cực đại với chi phí cực tiểu
    def __init__(self, N): # khai báo các mảng, biến ban đầu
        self.N = N
        self.add_flow = 0
        self.Max = 10**8
        self.size = N * 3 + 3
        self.par = [0] * self.size
        self.mark = [0] * self.size
        self.dis = [0] * self.size
        self.queue = collections.deque()

    def spfa(self, s, adj, rc): # Sử dụng SPFA (Shortest Path Faster Algorithm) để tìm đường tăng luồng có chi phí nhỏ nhất từ s đến đến điểm kết thúc
        for i in range(self.N * 2 + 2): # khởi tạo khoảng cách và đường đi các đỉnh
            self.mark[i] = 0
            self.dis[i] = self.Max

        self.dis[s] = 0 # khoảng cách từ đỉnh bắt đầu
        self.queue.append(s) # đưa vào trong queue
        self.mark[s] = 1 # đánh dấu điểm bắt đầu đang trong queue
        cur = None
        while self.queue: # khi còn điểm chưa xét xong
            cur = self.queue.popleft() # đưa ra khỏi hàng đợi
            self.mark[cur] = 0 # đánh dấu nút cur đã ra khổi hàng đợi
            for i in range(len(adj[cur])): # lấy những điểm kề với cur
                v, w = adj[cur][i] # v, w lần lượt là điểm kề và trọng số tới điểm kề đó
                if rc[cur][v] and self.dis[v] > self.dis[cur] + w: # nếu còn lưu lượng đi qua cur -> v và còn có thể tối ưu đường đi đến v
                    self.dis[v] = self.dis[cur] + w # cập nhật lại khoảng cách đến v
                    self.add_flow = min(self.add_flow, rc[cur][v]) # lấy lưu lượng nhỏ nhất
                    self.par[v] = cur # cập nhật lại cha của v là cur
                    if self.mark[v] == 0: # nếu điểm đó không trong hàng đợi
                        self.queue.append(v) # đưa vào hàng đợi để tối ưu những điểm kề v
                        self.mark[v] = 1 # đánh dấu lại v

    def solve(self, edge, adj, rc): # Thuật toán Dinic để ghép cặp cực đại
        res = 0
        t = self.N * 2 + 1 # điểm kết thúc là t

        while True: # trong khi còn có thể lấy lưu lượng lên
            for i in range(self.size): # khởi tạo lại cha các nút
                self.par[i] = -1
            self.par[0] = 0
            self.add_flow = self.Max # đặt lại lưu lượng
            self.spfa(0, adj, rc) # thực hiện spfa
            if self.par[t] == -1: # nếu không đi đến được nút kết thúc được
                break # kết thúc thuật toán
            v = t
            while v != 0: # truy vết lại đường đi từ nút kết thúc đến nút bắt đầu
                u = self.par[v] # lấy ra cha v
                res += edge[u][v] # cập nhật chi phí với trọng số của cạnh u -> v
                rc[v][u] += self.add_flow # cạnh v -> u tăng một lượng chứa 
                rc[u][v] -= self.add_flow # cạnh u -> mất đi lượng tương ứng
                v = u
        return res

def getDistance(curPosBox): # hàm lấy khoảng cách từ vị trí hộp hiện tại đến mọi đích sử dụng thuật toán bfs
    dr = [-1, 0, 1, 0] # khởi tạo mảng hướng đi
    dc = [0, -1, 0, 1]
    queue = collections.deque() # khai báo hàng chờ
    queue.append(curPosBox) # đưa vào vị trí của hộp
    mark = set() # khai báo set đánh dấu
    dis = {} # khai báo dict lưu khoảng cách
    mark.add(curPosBox) # đánh dấu vị trí bắt đầu
    dis[curPosBox] = 0 # đặt khoảng cách đến vị trí bắt đầu là 0
    while queue: # trong khi còn đỉnh chưa thăm
        pos = queue.popleft() # lấy đỉnh kế tiếp
        for i in range(4): # định 4 hướng đường đi
            newpos = (pos[0] + dr[i], pos[1] + dc[i]) # vị trí mới sau khi di chuyển
            if (newpos not in mark) and (newpos not in posWalls): # nếu vị trí mới chưa thăm và không phải là tường
                mark.add(newpos) # đánh dấu vị trí mới
                dis[newpos] = dis[pos] + 1 # cập nhật khoảng cách
                queue.append(newpos) # đưa vào trong hàng đợi
    mark.clear() # xóa tập đánh dấu
    return [dis[pos] for pos in posGoals] # trả về khoảng cách đến các điểm đích

def customHeuristic4(posPlayer, posBox): # Hàm heuristic tự thiết kế dựa theo việc mô hình hóa bài toán thành ghép cặp cực đại trong đồ thị 2 phía
    solve = MinimumCostMaxFlow(len(posBox)) # khai báo solver
    N = len(posBox) # lấy số đỉnh các hộp
    adj = [[] for _ in range(solve.size)] # khai báo mảng chứa cạnh kề
    edge = [[0] * (solve.size) for _ in range(solve.size)] # khai báo ma trận chứa trọng số
    rc = [[0] * (solve.size) for _ in range(solve.size)] # khai báo ma trận chứa lưu lượng

    """ 
    Ý tưởng: Mô hình hóa bài toán sang thành ghép cặp cực đại trong đồ thị 2 phía
    Với mỗi vị trí của cái hộp là một nút bên kia trong đồ thị,
            vị trí của từng đích đến là một nút phần còn lại trong đồ thị,
            -> Trở thành bài toán ghép cặp cực lại với trọng số mỗi cạnh thành khoảng cách độ đo giữa từng hộp với từng vị trí đích đến
    """
    for i in range(len(posBox)): # xét từng hộp
        dis = getDistance(posBox[i])
        for j in range(len(posGoals)): # xét từng vị trí đích
            adj[i + 1].append((j + N + 1, dis[j])) # nối cạnh hộp thứ i với đích đến thứ j
            adj[j + N + 1].append((i + 1, dis[j]))
            edge[i + 1][j + N + 1] = dis[j] # đặt chi phí giữa 2 đỉnh là khoảng cách
            edge[j + N + 1][i + 1] = -dis[j]
            rc[i + 1][j + N + 1] = solve.Max # khởi tạo lưu lượng
    
    # khởi tạo 2 đỉnh ảo bắt đầu và đích đến
    s = 0
    t = 2 * N + 1
    
    # nối 2 đỉnh ảo với các đỉnh 2 phía đồ thị
    for i in range(1, N + 1):
        adj[s].append((i, 0))
        adj[i].append((s, 0))
        rc[s][i] = 1
    
    for i in range(N + 1, 2 * N + 1):
        adj[t].append((i, 0))
        adj[i].append((t, 0))
        rc[i][t] = 1
    
    return solve.solve(edge, adj, rc) # trả về kết quả giải bài toán

def aStarSearch(gameState):
    """Implement aStarSearch approach"""
    start =  time.time()
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)
    temp = []
    start_state = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([start_state], heuristic(beginPlayer, beginBox))
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], heuristic(beginPlayer, start_state[1]))

    while len(frontier.Heap) > 0:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            break

        ### CONTINUE YOUR CODE FROM HERE
        if node[-1] not in exploredSet: # Nếu nút hiện tại xét chưa được xét
            exploredSet.add(node[-1]) # Đưa vào trong tập đã xét
            for action in legalActions(node[-1][0], node[-1][-1]): # lấy những hành động hợp lệ tại trạng thái hiện tại
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) # lấy ra vị trí mới của người chơi mới và vị trí hộp mới khi thực hiện hành động
                if isFailed(newPosBox): # nếu hành động có khả năng thất bại
                    continue # thì bỏ qua
                new_action = node_action + [action[-1]] # lấy hành động động mới đưa vào chuỗi
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(new_action[1:]) + customHeuristic4(newPosPlayer, newPosBox)) # đưa vào trong hàng đợi ưu tiên frontier nút mới và (chi phí đã tính + heuristic)
                actions.push(new_action, cost(new_action[1:]) + customHeuristic4(newPosPlayer, newPosBox)) # đưa vào trong hàng đợi ưu tiên action chuỗi hành động và (chi phí đã tính + heuristic)
    end =  time.time()
    print(end - start)
    print(len(exploredSet))
    return temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    print(layout)
    print(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':        
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    elif method == 'astar':
        result = aStarSearch(gameState)        
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    return result
