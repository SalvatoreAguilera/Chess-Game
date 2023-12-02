import pygame
import random
import sys
from itertools import combinations
import os
#changed

# current directory
dirname = os.path.dirname(__file__)

ORANGE = (255,185,15)
BLUE = (76, 252, 241)
WHITE = (255,255,255)
BLACK = (199,199,199)
WIDTH = 800
ROWS = 8

BKING= pygame.image.load(os.path.join(dirname, 'Pieces/bK.png'))
BBISHOP= pygame.image.load(os.path.join(dirname, 'Pieces/bB.png'))
BKNIGHT= pygame.image.load(os.path.join(dirname, 'Pieces/bN.png'))
BPAWN= pygame.image.load(os.path.join(dirname, 'Pieces/bP.png'))
BQUEEN= pygame.image.load(os.path.join(dirname, 'Pieces/bQ.png'))
BROOK= pygame.image.load(os.path.join(dirname, 'Pieces/bR.png'))

WKING= pygame.image.load(os.path.join(dirname, 'Pieces/wK.png'))
WBISHOP= pygame.image.load(os.path.join(dirname, 'Pieces/wB.png'))
WKNIGHT= pygame.image.load(os.path.join(dirname, 'Pieces/wN.png'))
WPAWN= pygame.image.load(os.path.join(dirname, 'Pieces/wP.png'))
WQUEEN= pygame.image.load(os.path.join(dirname, 'Pieces/wQ.png'))
WROOK= pygame.image.load(os.path.join(dirname, 'Pieces/wR.png'))


pygame.init()
WIN = pygame.display.set_mode((WIDTH,WIDTH))
pygame.display.set_caption('Chess')

priorMoves=[]
class Node:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = int(row * width)
        self.y = int(col * width)
        self.colour = WHITE
        self.piece = None

    def draw(self, WIN):
        pygame.draw.rect(WIN, self.colour, (self.x, self.y, WIDTH / ROWS, WIDTH / ROWS))
        if self.piece:
            WIN.blit(self.piece.image, (self.x, self.y))

class Piece:
    def __init__(self, team, piece_type,images):
        self.team=team
        self.image = images if self.team == 'B' else images
        self.type = piece_type

    def draw(self, x, y):
        WIN.blit(self.image, (x,y))
                   
def make_grid(rows, width):
    grid = []
    gap = width// rows
    #count = 0
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = Node(j,i, gap)
            if abs(i-j) % 2 == 0:
                node.colour=BLACK
            #PAWNS
            if i == 1:
                node.piece = Piece('W','PAWN',WPAWN)
            if i == 6:
                node.piece = Piece('B','PAWN',BPAWN)

            # ROOKS
            if i == 0 and j == 0 or j == 7 and i == 0:
                node.piece = Piece('W','ROOK', WROOK)
            if i == 7 and j == 0 or i == 7 and j == 7:
                node.piece = Piece('B','ROOK', BROOK)

            # KNIGHTS
            if i == 0 and j == 1 or j == 6 and i == 0:
                node.piece = Piece('W','KNIGHT', WKNIGHT)
            if i == 7 and j == 1 or i == 7 and j == 6:
                node.piece = Piece('B','KNIGHT', BKNIGHT)

            # BISHOP
            if i == 0 and j == 2 or j == 5 and i == 0:
                node.piece = Piece('W','BISHOP', WBISHOP)
            if i == 7 and j == 2 or i == 7 and j == 5:
                node.piece = Piece('B','BISHOP', BBISHOP)

            # KING
            if i == 0 and j == 4:
                node.piece = Piece('W','KING', WKING)
            if i == 7 and j == 4:
                node.piece = Piece('B','KING', BKING)
                
            # QUEEN
            if i == 0 and j == 3:
                node.piece = Piece('W','QUEEN', WQUEEN)
            if i == 7 and j == 3:
                node.piece = Piece('B','QUEEN', BQUEEN)
            grid[i].append(node)
    return grid

def draw_grid(win, rows, width):
    gap = width // ROWS
    for i in range(rows):
        pygame.draw.line(win, BLACK, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, BLACK, (j * gap, 0), (j * gap, width))
            
def update_display(win, grid, rows, width):
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    pygame.display.update()

def pawnMoves(col, row, grid):
    vectors = []

    if grid[col][row].piece and grid[col][row].piece.type == 'PAWN':
        if grid[col][row].piece.team == 'W' and col == 1:
            vectors = [[1, 0], [2, 0]]
            for vector in vectors:
                x,y = vector
                if grid[col+x][row+y].piece:
                    vectors.remove(vector) 
        elif grid[col][row].piece.team == 'B' and col == 6:
            vectors = [[-1, 0], [-2, 0]]
            for vector in vectors:
                x,y = vector
                if grid[col+x][row+y].piece:
                    vectors.remove(vector)
        elif grid[col][row].piece.team == 'W' and col != 1:
            if not grid[col+1][row].piece:
                vectors.append([1,0])
            if grid[col+1][row+1].piece:
                vectors.append([1,1])
            if grid[col+1][row-1].piece:
                vectors.append([1,-1])
                
        elif grid[col][row].piece.team == 'B' and col != 6:
            if not grid[col-1][row].piece:
                vectors.append([-1,0])
            if grid[col-1][row-1].piece:
                vectors.append([-1,-1])
            if grid[col-1][row+1].piece:
                vectors.append([-1,1])
    
    return vectors

def generatePotentialMoves(nodePosition, grid):
    checker = lambda x,y: x+y>=0 and x+y<8
    positions= []
    column, row = nodePosition
    if grid[column][row].piece:  
        if grid[column][row].piece.type == 'KING':
            vectors = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]]
        elif grid[column][row].piece.type == 'ROOK':
            vectors = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        elif grid[column][row].piece.type == 'KNIGHT':
            vectors = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
        elif grid[column][row].piece.type == 'BISHOP':
            vectors = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
        elif grid[column][row].piece.type == 'QUEEN':
            vectors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        elif grid[column][row].piece.type == 'PAWN':
            vectors = pawnMoves(column,row,grid)
            
            ''' and grid[column][row].piece.team == 'W' and column == 1:
            vectors = [[1,0],[2,0]]
        elif grid[column][row].piece.type == 'PAWN' and grid[column][row].piece.team == 'B' and column == 6:
            vectors = [[-1,0],[-2,0]]
        elif grid[column][row].piece.type == 'PAWN' and grid[column][row].piece.team == 'W' and column != 1:
            vectors = [[1,0],[1,1],[1,-1]]
        elif grid[column][row].piece.type == 'PAWN' and grid[column][row].piece.team == 'B' and column != 6:
            vectors = [[-1,0],[-1,-1],[-1,1]]   '''         
            
        for vector in vectors:
            columnVector, rowVector = vector
            #OUT OF BOUNDS CHECK
            if checker(columnVector,column) and checker(rowVector,row):
                if not grid[(column+columnVector)][(row+rowVector)].piece:
                    positions.append((column+columnVector,row+rowVector))
                elif grid[column+columnVector][row+rowVector].piece and\
                    grid[column+columnVector][row+rowVector].piece.team==opposite(grid[column][row].piece.team):
                    positions.append((columnVector+ column,rowVector+ row))
                    
                
                
                        
    return positions

def HighlightpotentialMoves(piecePosition, grid):
    positions = generatePotentialMoves(piecePosition, grid)
    for position in positions:
        Column,Row = position
        grid[Column][Row].colour=BLUE
        
def highlight(ClickedNode, Grid, OldHighlight):
    Column,Row = ClickedNode
    Grid[Column][Row].colour=ORANGE
    if OldHighlight:
        resetColours(Grid, OldHighlight)
    HighlightpotentialMoves(ClickedNode, Grid)
    return (Column,Row)

def opposite(team):
    return "W" if team=="B" else "B"

def getNode(rows, width):
    gap = width//rows
    RowX,RowY = pygame.mouse.get_pos()
    Row = RowX//gap
    Col = RowY//gap
    return (Col,Row)

def resetColours(grid, node):
    positions = generatePotentialMoves(node, grid)
    positions.append(node)

    for colouredNodes in positions:
        nodeX, nodeY = colouredNodes
        grid[nodeX][nodeY].colour = BLACK if abs(nodeX - nodeY) % 2 == 0 else WHITE

def move(grid, piecePosition, newPosition):
    #resetColours(grid, piecePosition)
    newColumn, newRow = newPosition
    oldColumn, oldRow = piecePosition

    piece = grid[oldColumn][oldRow].piece
    grid[newColumn][newRow].piece=piece
    grid[oldColumn][oldRow].piece = None

    return opposite(piece.team) 
 
def main(WIDTH,ROWS):
    grid = make_grid(ROWS, WIDTH)
    highlightedPiece = None
    currMove = 'B'
    while True:
        for event in pygame.event.get():
            if event.type== pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                clickedNode = getNode(ROWS, WIDTH)
                ClickedPositionColumn, ClickedPositionRow = clickedNode
                if grid[ClickedPositionColumn][ClickedPositionRow].colour == BLUE:
                    if highlightedPiece:
                        pieceColumn, pieceRow = highlightedPiece
                    if currMove == grid[pieceColumn][pieceRow].piece.team:
                            resetColours(grid, highlightedPiece)
                            currMove=move(grid, highlightedPiece, clickedNode)
                elif highlightedPiece == clickedNode:
                        pass
                else:
                    if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                        if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                            highlightedPiece = highlight(clickedNode, grid, highlightedPiece)
                
            update_display(WIN,grid,ROWS,WIDTH)
main(WIDTH, ROWS)