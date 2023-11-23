#Class Project pygame chess
#members: Jonathan Morley, , ,
#Version 1.1 "en passant update"
#11/21/23 not done at 2:17 A.M.
#

import pygame
#import random
import sys
#from itertools import combinations
import os

# current directory
dirname = os.path.dirname(__file__)

ORANGE = (255,185,15)
BLUE = (76, 252, 241)
WHITE = (255,255,255)
BLACK = (199,199,199)
WIDTH = 800
ROWS = 8


#pieces are global variables, able to be called by all subsequent classes, functions, and methods.
#IMPORTANT TO NOT NAME LOCAL VARIABLES AS GLOBAL OR TRY TO GIVE T
BKING = pygame.image.load(os.path.join(dirname, 'Pieces/bK.png'))
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
        # The constructor for the Node class
        # initializes a Node with the specified row, column, and width.
        self.row = row
        self.col = col
        # Calculate the x and y coordinates for the Node.
        # The x coordinate is multiplying the row index by the width of each cell.
        # The y coordinate is multiplying the column index by the width of each cell.
        self.x = int(row * width)
        self.y = int(col * width)
        self.colour = WHITE  # Set the default colour of the Node to white.
        self.piece = None    # Initialize the piece attribute to None. This can later be set to a chess piece.

    def draw(self, WIN):
        # The draw method is responsible for drawing the Node on the screen.
        # It takes a Pygame window (WIN) as an argument.
         # Draw a rectangle on the WIN to represent the Node.
        # The rectangle's position and size are based on the Node's x, y coordinates and the width of each cell.
        pygame.draw.rect(WIN, self.colour, (self.x, self.y, WIDTH / ROWS, WIDTH / ROWS))
        # If the Node has a chess piece, draw the piece's image at the Node's position.
        if self.piece:
            WIN.blit(self.piece.image, (self.x, self.y))

class Piece:
    def __init__(self, team, piece_type,images):
        #assigns team
        self.team=team
        #default color is white, if statement sets opposite team to black
        self.image = images if self.team == 'B' else images
        #assigns attribute of the piece type, pawn, rook, king, etc
        self.type = piece_type
        self.hasMoved = False  # if pieces havent moved, special rules may apply

    def draw(self, x, y):
        WIN.blit(self.image, (x,y))

class LastMove:
    def __init__(self, piece, start_pos, end_pos, move_type=None):
        self.piece = piece        # The piece that was moved
        self.start_pos = start_pos # The starting position of the move (tuple: (col, row))
        self.end_pos = end_pos     # The ending position of the move (tuple: (col, row))
        self.move_type = move_type # Type of move (e.g., "en_passant", "castle", "two_step_pawn", etc.)
def make_grid(rows, width):
    #initialize grid array
    grid = []
    #The double slash // in Python is used for floor division.
    #It divides the left-hand operand by the right-hand operand and rounds the result down to the nearest whole number.
    #interesting, must remember for later
    gap = width // rows

    # Iterate over each row.
    for i in range(rows):
        # Add an empty list to represent a new row in the grid.
        grid.append([])
        # Iterate over each column in the current row.
        for j in range(rows):
            # Create a new Node object for each cell in the grid.
            node = Node(j,i, gap)
            # Set the node color to black on alternating cells.
            if abs(i-j) % 2 == 0:
                node.colour=BLACK
            # Initialize pawns on the 2nd and 7th rows.
            if i == 1:
                node.piece = Piece('W','PAWN',WPAWN)
            if i == 6:
                node.piece = Piece('B','PAWN',BPAWN)

            #  Place rooks in the corners of the board.
            if i == 0 and j == 0 or j == 7 and i == 0:
                node.piece = Piece('W','ROOK', WROOK)
            if i == 7 and j == 0 or i == 7 and j == 7:
                node.piece = Piece('B','ROOK', BROOK)

            # Initialize knights next to the rooks.
            if i == 0 and j == 1 or j == 6 and i == 0:
                node.piece = Piece('W','KNIGHT', WKNIGHT)
            if i == 7 and j == 1 or i == 7 and j == 6:
                node.piece = Piece('B','KNIGHT', BKNIGHT)

            # Place bishops next to the knights.
            if i == 0 and j == 2 or j == 5 and i == 0:
                node.piece = Piece('W','BISHOP', WBISHOP)
            if i == 7 and j == 2 or i == 7 and j == 5:
                node.piece = Piece('B','BISHOP', BBISHOP)

            #  Set the king and queen in the middle of the first and last rows.
            if i == 0 and j == 4:
                node.piece = Piece('W','KING', WKING)
            if i == 7 and j == 4:
                node.piece = Piece('B','KING', BKING)
            if i == 0 and j == 3:
                node.piece = Piece('W','QUEEN', WQUEEN)
            if i == 7 and j == 3:
                node.piece = Piece('B','QUEEN', BQUEEN)

            # Add the node to the current row in the grid.
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

def pawnMoves(col, row, grid, lastMove):
    vectors = []
    team = grid[col][row].piece.team

    # Forward movement for pawns
    direction = -1 if team == 'B' else 1
    start_row = 6 if team == 'B' else 1
    enemy_start_row = 3 if team == 'B' else 4

    # Standard pawn moves
    if not grid[col + direction][row].piece:
        vectors.append([direction, 0])
        if col == start_row and not grid[col + 2 * direction][row].piece:
            vectors.append([2 * direction, 0])

    # Pawn captures
    for dx in [-1, 1]:
        if row + dx < 0 or row + dx >= ROWS:
            continue
        if grid[col + direction][row + dx].piece and grid[col + direction][row + dx].piece.team != team:
            vectors.append([direction, dx])

    # En passant
    if lastMove and lastMove.piece.type == 'PAWN' and abs(lastMove.start_pos[0] - lastMove.end_pos[0]) == 2:
        if lastMove.end_pos[0] == col and abs(lastMove.end_pos[1] - row) == 1:
            if lastMove.end_pos[1] == row + 1 or lastMove.end_pos[1] == row - 1:
                if lastMove.piece.team != team and lastMove.end_pos[0] == enemy_start_row:
                    en_passant_capture = [direction, lastMove.end_pos[1] - row]
                    vectors.append(en_passant_capture)
    return vectors

def rookMoves(col, row, grid):
    vectors = []
    for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]: #four directions, up down left right
        for distance in range(1,ROWS):
            newColumn, newRow = col + distance * dx, row + distance * dy

            if not 0 <= newColumn < ROWS or not 0 <= newRow < ROWS:
                break #out of bounds checking

            if grid [newColumn][newRow].piece:
                # If there is a piece in the way, stop looking further in this direction
                if grid[newColumn][newRow].piece.team != grid[col][row].piece.team:
                    vectors.append([distance * dx, distance * dy])  # Can capture opponent's piece
                break

            vectors.append([distance * dx, distance * dy])  # Empty square, add to possible moves
    return vectors

def bishopMoves(col, row, grid):
    vectors = []
    for dx, dy in [[1, 1], [1, -1], [-1, 1], [-1, -1]]: #diagonal directions
        for distance in range (1, ROWS):
            newColumn, newRow = col + distance * dx, row + distance * dy
            if not 0 <= newColumn < ROWS or not 0 <= newRow < ROWS:
                break #out of bounds checking
            if grid [newColumn][newRow].piece:
                # If there is a piece in the way, stop looking further in this direction
                if grid[newColumn][newRow].piece.team != grid[col][row].piece.team:
                    vectors.append([distance * dx, distance * dy])  # Can capture opponent's piece
                break
            vectors.append([distance * dx, distance * dy])  # Empty square, add to possible moves
    return vectors
def queenMoves(col, row, grid):
    vectors = []
    for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]: #diagonal directions
        for distance in range (1, ROWS):
            newColumn, newRow = col + distance * dx, row + distance * dy
            if not 0 <= newColumn < ROWS or not 0 <= newRow < ROWS:
                break #out of bounds checking
            if grid [newColumn][newRow].piece:
                # If there is a piece in the way, stop looking further in this direction
                if grid[newColumn][newRow].piece.team != grid[col][row].piece.team:
                    vectors.append([distance * dx, distance * dy])  # Can capture opponent's piece
                break
            vectors.append([distance * dx, distance * dy])  # Empty square, add to possible moves
    return vectors

def canCastle(kingPos, rookPos, grid, currMove):
    kingRow, kingCol = kingPos
    rookRow, rookCol = rookPos

    # Check if the king and the rook have not moved
    if grid[kingRow][kingCol].piece.hasMoved or grid[rookRow][rookCol].piece.hasMoved:
        return False

    # Check for pieces between the king and the rook
    start = min(kingCol, rookCol) + 1
    end = max(kingCol, rookCol)
    for col in range(start, end):
        if grid[kingRow][col].piece is not None:
            return False

    # Check if the king is in check
    if isKingInCheck(currMove, grid):
        return False

    # Check if the king passes through or lands on a square that is under attack
    direction = 1 if kingCol < rookCol else -1
    for i in range(1, 3):
        if isSquareUnderAttack((kingRow, kingCol + i * direction), grid, currMove):
            return False

    return True

def isSquareUnderAttack(pos, grid, currMove):
    for i in range(ROWS):
        for j in range(ROWS):
            if grid[i][j].piece and grid[i][j].piece.team != currMove:
                if pos in generatePotentialMovesWithoutCastling((i, j), grid):
                    return True
    return False

def print_grid_positions(grid):
    for row in grid:
        for node in row:
            if node.piece:
                print(f"{node.piece.type} at ({node.row}, {node.col}) - Team: {node.piece.team}")

def move(grid, piecePosition, newPosition, lastMove):
    #resetColours(grid, piecePosition)
    # Extract the column and row from the newPosition
    newColumn, newRow = newPosition
    # Extract the column and row from the piecePosition
    oldColumn, oldRow = piecePosition
    # Retrieve the piece object from the old position on the grid
    piece = grid[oldColumn][oldRow].piece

    if piece is None:
        print("Move failed: No piece at the starting position")
        return False  # Cannot move a piece that doesn't exist
    # Check for Castling
    if piece.type == 'KING' and abs(newRow - oldRow) == 2:
        # Determine if it's a kingside or queenside castle
        isKingside = newRow > oldRow
        rookOldCol = 7 if isKingside else 0
        rookNewCol = 5 if isKingside else 3  # Rook's new position in castling

        # Move the Rook
        grid[oldColumn][rookNewCol].piece = grid[oldColumn][rookOldCol].piece
        grid[oldColumn][rookOldCol].piece = None
        grid[oldColumn][rookNewCol].piece.hasMoved = True

    # Simulate the move
    simulatedGrid = simulateMove(oldColumn, oldRow, newColumn, newRow, grid)

    # Check if the simulated move puts your own king in check
    if isKingInCheck(piece.team, simulatedGrid):
        # Illegal move, puts own king in check, so revert the move
        print("Move failed: Move puts own king in check")
        return False  # Return a value that indicates the move was not successful

    # Check for en passant
    if piece.type == 'PAWN' and newColumn != oldColumn and not grid[newColumn][newRow].piece:
        print("En passant attempt detected")
        # This is a diagonal move without a normal capture, potential en passant
        if lastMove and lastMove.piece.type == 'PAWN' and abs(lastMove.start_pos[0] - lastMove.end_pos[0]) == 2:
            if lastMove.end_pos[1] == newRow and abs(lastMove.end_pos[1] - oldRow) == 1:
                # Remove the pawn that was "passed over" during en passant
                passedPawnRow = lastMove.start_pos[0] if piece.team == 'W' else lastMove.end_pos[0]
                grid[passedPawnRow][newRow].piece = None

    # Regular Move
    grid[newColumn][newRow].piece = piece
    grid[oldColumn][oldRow].piece = None
    piece.hasMoved = True

    print(f"Move completed from {piecePosition} to {newPosition}")
    return opposite(piece.team)

def generatePotentialMoves(nodePosition, grid, lastMove):
    checker = lambda x, y: x + y >=0 and x + y < 8
    positions = []
    column, row = nodePosition
    piece = grid[column][row].piece

    if piece.type == 'KING':
        # Standard king moves
        vectors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

        # Check for castling (if the king has not moved)
        if not piece.hasMoved:
            # Check both sides for castling
            for rookCol in [0, 7]:
                if grid[column][rookCol].piece and grid[column][rookCol].piece.type == 'ROOK' and not grid[column][rookCol].piece.hasMoved:
                    if canCastle((column, row), (column, rookCol), grid, piece.team):
                        castlingMove = (column, 2 if rookCol == 0 else 6)
                        positions.append(castlingMove)
    elif piece.type == 'ROOK':
        vectors = rookMoves(column, row, grid)
    elif piece.type == 'KNIGHT':
        vectors = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
    elif piece.type == 'BISHOP':
        vectors = bishopMoves(column, row, grid)
    elif piece.type == 'QUEEN':
        vectors = queenMoves(column, row, grid)
    elif piece.type == 'PAWN':
        vectors = pawnMoves(column, row, grid, lastMove)

    for vector in vectors:
        columnVector, rowVector = vector
        if checker(columnVector, column) and checker(rowVector, row):
            newPos = (column + columnVector, row + rowVector)
            if not grid[newPos[0]][newPos[1]].piece:
                positions.append(newPos)
                #print(f"Adding empty move: {newPos}")
            elif grid[newPos[0]][newPos[1]].piece.team == opposite(piece.team):
                positions.append(newPos)
                #print(f"Adding capture move: {newPos}")

    #print(f"Generated {len(positions)} potential moves for {piece.type} at {nodePosition}")
    return positions

#I needed to create this because when checking if castling was possible, my original method
#called generatePotentialMoves within isKingInCheck and it would recursively call itself
#either out of bounds or infinitely causing the program to crash.
#probably not the most efficient but it worked
def generatePotentialMovesWithoutCastling(nodePosition, grid):
    # This function is similar to generatePotentialMoves but does not check for castling moves.
    checker = lambda x, y: 0 <= x < ROWS and 0 <= y < ROWS
    positions = []
    column, row = nodePosition
    piece = grid[column][row].piece

    if not piece:
        return positions  # Return empty list if there's no piece at the given position

    # Define move vectors for different pieces
    if piece.type == 'ROOK':
        vectors = rookMoves(column, row, grid)
    elif piece.type == 'KNIGHT':
        vectors = [[1, 2], [1, -2], [-1, 2], [-1, -2], [2, 1], [2, -1], [-2, 1], [-2, -1]]
    elif piece.type == 'BISHOP':
        vectors = bishopMoves(column, row, grid)
    elif piece.type == 'QUEEN':
        vectors = queenMoves(column, row, grid)
    elif piece.type == 'PAWN':
        vectors = pawnMoves(column, row, grid, None)  # Pass None for lastMove to avoid en passant
    elif piece.type == 'KING':
        vectors = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]

    # Check each move vector and add it to positions if it's valid
    for vector in vectors:
        columnVector, rowVector = vector
        newPos = (column + columnVector, row + rowVector)
        if checker(newPos[0], newPos[1]):
            # Add the new position if it's on the board and either empty or occupied by an opposing piece
            if not grid[newPos[0]][newPos[1]].piece or grid[newPos[0]][newPos[1]].piece.team != piece.team:
                positions.append(newPos)

    return positions



def HighlightpotentialMoves(piecePosition, grid,lastMove):
    #takes position of a chess piece (piecePosition) and the board grid
    #call generatePotentialMoves to get a list of all valid move positions of the piece
    #piecePosition is the current location of the piece, and grid is the current state of the chessboard
    positions = generatePotentialMoves(piecePosition, grid, lastMove)
    # Iterate through each position in the list of potential moves.
    for position in positions:
        # Unpack the position tuple into Column and Row variables.
        Column,Row = position
        # Change the color of the grid cell at each potential move position to blue.
        grid[Column][Row].colour=BLUE

def highlight(ClickedNode, Grid, OldHighlight, lastMove):
    # This function takes three parameters:
    # ClickedNode: The position (column, row) of the node that was clicked or selected.
    # Grid: A 2D array representing the chessboard, where each element is a Node object.
    # OldHighlight: The position of the previously highlighted node, if any.

    # Extract the column and row from the ClickedNode tuple.
    Column,Row = ClickedNode
    # Set the colour of the clicked node to orange, visually indicating selection.
    Grid[Column][Row].colour=ORANGE
    # If there was a previously highlighted node (OldHighlight is not None),
    # reset the colours of the nodes that were highlighted before.
    # This is likely done to remove the previous move highlights from the grid.
    if OldHighlight:
        resetColours(Grid, OldHighlight, lastMove)
    # Highlight all potential moves for the piece at the clicked node.
    # This function changes the colour of all nodes where the piece can legally move.
    HighlightpotentialMoves(ClickedNode, Grid, lastMove)
    # Return the position of the clicked node.
    # This can be used to track the currently highlighted node.
    return (Column,Row)

def opposite(team):
    return "W" if team=="B" else "B"

def getNode(rows, width):
    # This function is designed to translate a
    # mouse click position into grid coordinates on a chessboard.
    gap = width // rows
    # Get the current mouse position.
    RowX, RowY = pygame.mouse.get_pos()
    # The division by 'gap' translates the pixel position to the grid position.
    Row = RowX // gap
    Col = RowY // gap
    # Return a tuple (Col, Row) representing the grid coordinates of the mouse click.
    # This corresponds to the column and row in the chessboard grid.
    return (Col,Row)

def resetColours(grid, lastHighlight, lastMove):
    # Iterate over the entire grid to reset colors
    for i in range(ROWS):
        for j in range(ROWS):
            # Set the color based on the position to get the checkerboard pattern
            if (i + j) % 2 == 0:
                grid[i][j].colour = WHITE
            else:
                grid[i][j].colour = BLACK

    #for debugging but can be implmented in real game if we wnat
    # If there is a last move, you might want to highlight the last move made
    if lastMove:
        # Highlight the start and end position of the last move
        startRow, startCol = lastMove.start_pos
        endRow, endCol = lastMove.end_pos
        grid[startRow][startCol].colour = ORANGE  # Use a different color if needed
        grid[endRow][endCol].colour = ORANGE  # Use a different color if needed



def canMoveOutOfCheck(currMove, grid):
    # Iterate through all pieces of the current player
    print(f"Checking if {currMove} can move out of check")
    for row in range(ROWS):
        for col in range(ROWS):
            node = grid[row][col]
            if node.piece and node.piece.team == currMove:
                print(f"Analyzing moves for {node.piece.type} at {(row, col)}")
                # Generate all potential moves for this piece
                potentialMoves = generatePotentialMoves((row, col), grid, None)

                # Check each potential move
                for move in potentialMoves:
                    # Simulate the move
                    simulatedGrid = simulateMove(row, col, move[0], move[1], grid)
                    print(f"Simulating move from {(row, col)} to {move}")

                    # Check if the king is still in check after this move
                    if not isKingInCheck(currMove, simulatedGrid):
                        print(f"Move from {(row, col)} to {move} gets the king out of check")
                        return True  # Found a move that can get the king out of check
    print("No moves found to get the king out of check")
    return False  # No move found to get the king out of check

def simulateMove(fromRow, fromCol, toRow, toCol, grid):
    # Manually create a deep copy of the grid
    simulatedGrid = []
    for row in grid:
        newRow = []
        for node in row:
            # Copy the Node object
            copiedNode = Node(node.row, node.col, WIDTH // ROWS)
            copiedNode.colour = node.colour

            # Copy the Piece object if it exists
            if node.piece:
                copiedPiece = Piece(node.piece.team, node.piece.type, node.piece.image)
                copiedNode.piece = copiedPiece

            newRow.append(copiedNode)
        simulatedGrid.append(newRow)

    # Move the piece in the simulated grid
    simulatedGrid[toRow][toCol].piece = simulatedGrid[fromRow][fromCol].piece
    simulatedGrid[fromRow][fromCol].piece = None

    return simulatedGrid

def isKingInCheck(currMove, grid):
    # currMove is the current player's turn ('W' or 'B')
    # Find the king's position
    kingPos = None
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            node = grid[i][j]
            if node.piece and node.piece.type == 'KING' and node.piece.team == currMove:
                kingPos = (i, j)
                break
        if kingPos:
            break

    if not kingPos:
        return False  # King not found, which is an error condition

    # Check if any of the opponent's pieces can attack the king
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j].piece and grid[i][j].piece.team != currMove:
                # Check the potential moves without generating castling moves to avoid recursion
                potentialMoves = generatePotentialMovesWithoutCastling((i, j), grid)
                if kingPos in potentialMoves:
                    return True  # King is in check

    return False
'''
#working check
    kingPos = None
    for i in range(ROWS):
        for j in range(ROWS):
            node = grid[i][j]
            if node.piece and node.piece.type == 'KING' and node.piece.team == currMove:
                kingPos = (i, j)
                break
        if kingPos:
            break

    if kingPos is None:
        return False  # King not found, which is an error condition

    # Check if any of the opponent's pieces can attack the king
    for i in range(ROWS):
        for j in range(ROWS):
            if grid[i][j].piece and grid[i][j].piece.team != currMove:
                if kingPos in generatePotentialMoves((i, j), grid, None):
                    return True  # King is in check
'''
def isKingInCheckmate(currMove, grid):
    # First, confirm the king is currently in check
    if not isKingInCheck(currMove, grid):
        return False  # If the king is not in check, it can't be checkmate

    # Find the king's position
    kingPos = None
    for row in range(ROWS):
        for col in range(ROWS):
            node = grid[row][col]
            if node.piece and node.piece.type == 'KING' and node.piece.team == currMove:
                kingPos = (row, col)
                break
        if kingPos:
            break

    if kingPos is None:
        raise Exception("King not found, which should never happen")

    # Try all possible moves for all pieces of the current player
    for row in range(ROWS):
        for col in range(ROWS):
            node = grid[row][col]
            if node.piece and node.piece.team == currMove:
                potentialMoves = generatePotentialMoves((row, col), grid, None)
                for move in potentialMoves:
                    # Simulate each move
                    simulatedGrid = simulateMove(row, col, move[0], move[1], grid)
                    # If after any move, the king is not in check, it's not checkmate
                    if not isKingInCheck(currMove, simulatedGrid):
                        return False

    # If no move takes the king out of check, it's checkmate
    return True

def main(WIDTH,ROWS):
    # Initialize the chess grid with nodes and pieces
    grid = make_grid(ROWS, WIDTH)
    # Track the currently highlighted piece, initially set to None
    highlightedPiece = None
    # Set the current move to 'B' (Black) as the starting player
    currMove = 'B'
    lastMove = None  # Initialize lastMove as None
    # Start the game loop
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                clickedNode = getNode(ROWS, WIDTH)
                ClickedPositionColumn, ClickedPositionRow = clickedNode
                # Check if a valid move is made (blue colored cell)
                if grid[ClickedPositionColumn][ClickedPositionRow].colour == BLUE:
                    if highlightedPiece:
                        pieceColumn, pieceRow = highlightedPiece
                    if currMove == grid[pieceColumn][pieceRow].piece.team:
                        # Try to execute the move
                        moveSuccessful = move(grid, highlightedPiece, clickedNode, lastMove)
                        if moveSuccessful:
                            # Move was successful, update lastMove, switch turns, etc.
                            # Reset colors of the previous move
                            resetColours(grid, highlightedPiece,lastMove)
                            # Update lastMove object with new move details
                            movedPiece = grid[ClickedPositionColumn][ClickedPositionRow].piece
                            lastMove = LastMove(movedPiece, highlightedPiece, clickedNode)
                            highlightedPiece = None  # Reset the highlighted piece so no cell is marked as selected
                            # Add additional details if necessary, possibly move type?
                            # After a move is made, check if the king is in check
                            if isKingInCheck(opposite(currMove), grid):
                                print("Check!")  # Notify the player
                                # Determine if the opponent can move out of check
                                if not canMoveOutOfCheck(opposite(currMove), grid):
                                    if isKingInCheckmate(opposite(currMove), grid):
                                        print("Checkmate!")
                                    else:
                                        print("Stalemate!")
                                # Switch turns
                            currMove = opposite(currMove)
                        else:
                            # Move was not successful (would put/leave king in check), handle accordingly
                            print("Illegal move: would put/leave king in check.")
                            # Do not switch turns, allow the player to make another move

                            # Switch turns
                            #currMove = 'W' if currMove == 'B' else 'B'
                elif highlightedPiece == clickedNode:
                        pass
                else:
                    if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                        if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                            highlightedPiece = highlight(clickedNode, grid, highlightedPiece, lastMove)

            update_display(WIN,grid,ROWS,WIDTH)
main(WIDTH, ROWS)
