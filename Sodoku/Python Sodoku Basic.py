
# coding: utf-8

# In[11]:


grid = [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,7,9]]


# In[12]:


def print_board():
    global grid
    for i in list(range(len(grid))):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - ")
            
        for j in list(range(len(grid[0]))):
            if j % 3 == 0 and j != 0:
                print(" | ", end = "")
        
            if j == 8:
                print(grid[i][j])
            else:
                print(str(grid[i][j]) + " ", end = "")
print_board()


# In[13]:


def possible(y,x,n):
    global grid
    for i in list(range(9)):
        if grid[y][i] == n:
            return False
    for i in list(range(9)):
        if grid[i][x] == n:
            return False
    x0 = (x//3)*3
    y0 = (y//3)*3
    for i in range(0,3):
        for j in range(0,3):
            if grid[y0+i][x0+j] == n:
                return False
    return True


# In[14]:


def solve():
    global grid
    for y in list(range(9)):
        for x in list(range(9)):
            if grid[y][x] == 0:
                for n in range(1,10):
                    if possible(y,x,n):
                        grid[y][x] = n
                        solve()
                        grid[y][x] = 0
                return
    print_board()
    input("More?")


# In[15]:


solve()


# In[21]:


#grid = [[5,3,0,0,7,0,0,0,0],[6,0,0,1,9,5,0,0,0],[0,9,8,0,0,0,0,6,0],[8,0,0,0,6,0,0,0,3],[4,0,0,8,0,3,0,0,1],[7,0,0,0,2,0,0,0,6],[0,6,0,0,0,0,2,8,0],[0,0,0,4,1,9,0,0,5],[0,0,0,0,8,0,0,0,0]]
#solve()


# In[26]:


def generate_valid_9x9():
    board = [[0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0,0]]
    print(board)
generate_valid_9x9()

