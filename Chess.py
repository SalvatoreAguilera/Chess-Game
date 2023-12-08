#Class Project pygame chess
#members: Jonathan Morley, , ,
#Version 1.3 "broken en passant and pawn promotion"
#11/27/23 not done at 2:17 A.M.
#

import pygame
import sys
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

#for checkers
RED= pygame.image.load(os.path.join(dirname, 'images/red.png'))
GREEN= pygame.image.load(os.path.join(dirname, 'images/green.png'))
REDKING = pygame.image.load(os.path.join(dirname, 'images/redking.png'))
GREENKING = pygame.image.load(os.path.join(dirname, 'images/greenking.png'))

CHECKERSWHITE = (255,255,255)
CHECKERSBLACK = (0,0,0)
CHECKERSORANGE = (235, 168, 52)
CHECKERSBLUE = (76, 252, 241)

pygame.init()
WIN = pygame.display.set_mode((WIDTH,WIDTH))


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
        self.image = images# if self.team == 'B' else images
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
            if i == 6:
                node.piece = Piece('W','PAWN',WPAWN)
            if i == 1:
                node.piece = Piece('B','PAWN',BPAWN)

            #  Place rooks in the corners of the board.
            if i == 7 and j == 0 or i == 7 and j == 7:
                node.piece = Piece('W','ROOK', WROOK)
            if i == 0 and j == 0 or j == 7 and i == 0:
                node.piece = Piece('B','ROOK', BROOK)

            # Initialize knights next to the rooks.
            if i == 7 and j == 1 or i == 7 and j == 6:
                node.piece = Piece('W','KNIGHT', WKNIGHT)
            if i == 0 and j == 1 or j == 6 and i == 0:
                node.piece = Piece('B','KNIGHT', BKNIGHT)

            # Place bishops next to the knights.
            if i == 7 and j == 2 or i == 7 and j == 5:
                node.piece = Piece('W','BISHOP', WBISHOP)
            if i == 0 and j == 2 or j == 5 and i == 0:
                node.piece = Piece('B','BISHOP', BBISHOP)

            #  Set the king and queen in the middle of the first and last rows.
            if i == 7 and j == 4:
                node.piece = Piece('W','KING', WKING)
            if i == 0 and j == 4:
                node.piece = Piece('B','KING', BKING)
            if i == 7 and j == 3:
                node.piece = Piece('W','QUEEN', WQUEEN)
            if i == 0 and j == 3:
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

def draw_pause(window):
    # Create a surface to cover the screen
    overlay = pygame.Surface(window.get_size(), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))
    window.blit(overlay, (0, 0))

    # Define button dimensions and positions
    button_width, button_height = 200, 50
    x = (window.get_width() - button_width) // 2
    y_start = (window.get_height() - 3 * button_height) // 2

    # button texts
    button_texts = ["To Menu", "Restart", "Quit"]

    # create and draw the rectangles
    menu_button = pygame.Rect(x, y_start, button_width, button_height)
    restart_button = pygame.Rect(x, y_start + button_height + 10, button_width, button_height)
    quit_button  = pygame.Rect(x, y_start + 2 * (button_height + 10), button_width, button_height)
    # Load font
    font = pygame.font.SysFont(None, 36)

    for i, rect in enumerate([menu_button, restart_button, quit_button]):
        pygame.draw.rect(window, (255, 255, 255), rect)  # White color for the buttons
        text_surface = font.render(button_texts[i], True, (0, 0, 0))  # Black color for the text
        text_rect = text_surface.get_rect(center=rect.center)
        window.blit(text_surface, text_rect)


    # Return the rectangles
    return [quit_button, restart_button, menu_button]

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
    direction = -1 if team == 'W' else 1
    start_row = 6 if team == 'W' else 1
    enemy_start_row = 3 if team == 'W' else 4

    # Ensure pawn does not move forward if at the end of the board
    if (team == 'B' and col == 7) or (team == 'W' and col == 0):
        return vectors  # No forward moves possible

    # Standard pawn moves
    if not grid[col + direction][row].piece:
        vectors.append([direction, 0])
        if col == start_row and not grid[col + 2 * direction][row].piece:
            vectors.append([2 * direction, 0])

    # Pawn captures, including en passant
    for dx in [-1, 1]:
        if 0 <= row + dx < ROWS:
            # Regular capture
            if grid[col + direction][row + dx].piece and grid[col + direction][row + dx].piece.team != team:
                vectors.append([direction, dx])
            # En passant
            if col == enemy_start_row:  # Pawn must be on the fifth rank to perform en passant
                if lastMove and lastMove.piece.type == 'PAWN' and abs(lastMove.start_pos[0] - lastMove.end_pos[0]) == 2:
                    if lastMove.end_pos[1] == row + dx:  # The last move's pawn must be adjacent to the current pawn
                        en_passant_capture = [direction, dx]
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
        # This is a diagonal move without a normal capture, potential en passant

        if lastMove and lastMove.piece.type == 'PAWN' and abs(lastMove.start_pos[0] - lastMove.end_pos[0]) == 2:
            if lastMove.end_pos[1] == newRow and abs(lastMove.end_pos[1] - oldRow) == 1:
                # Remove the pawn that was "passed over" during en passant
                passedPawnRow = lastMove.end_pos[0]#lastMove.start_pos[0] if piece.team == 'B' else lastMove.end_pos[0]
                grid[passedPawnRow][newRow].piece = None

    # Regular Move
    grid[newColumn][newRow].piece = piece
    grid[oldColumn][oldRow].piece = None
    piece.hasMoved = True

    # Promotion check for pawns reaching the opposite end of the board
    if piece.type == 'PAWN':
        if (piece.team == 'W' and newColumn == 0) or (piece.team == 'B' and newColumn == 7):
            print("Pawn promotion triggered")
            promotePawn(grid, newPosition, piece.team)  # Call the promotion function

    print(f"Move completed from {piecePosition} to {newPosition}")
    return True

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
    if RowX < 0 or RowX >= width or RowY < 0 or RowY >= width:
        return None  # Click is outside the board
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
                grid[i][j].colour = BLACK
            else:
                grid[i][j].colour = WHITE

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

############ START PAWN PROMOTION ############

def promotePawn(grid, pawnPos, team):
    # Display promotion options
    print("Promote Pawn function called")
    promotion_options = displayPromotionOptions(team)
    # Wait for player input and get the selected piece
    selectedPieceType = getPlayerChoice(promotion_options)
    # Replace the pawn with the new piece
    print(f"Selected piece for promotion: {selectedPieceType}")
    new_piece = None
    if selectedPieceType == 'queen':
        new_piece = Piece(team, 'QUEEN', WQUEEN if team == 'W' else BQUEEN)
    elif selectedPieceType == 'rook':
        new_piece = Piece(team, 'ROOK', WROOK if team == 'W' else BROOK)
    elif selectedPieceType == 'knight':
        new_piece = Piece(team, 'KNIGHT', WKNIGHT if team == 'W' else BKNIGHT)
    elif selectedPieceType == 'bishop':
        new_piece = Piece(team, 'BISHOP', WBISHOP if team == 'W' else BBISHOP)
    if new_piece:
        grid[pawnPos[0]][pawnPos[1]].piece = new_piece
        print(f"Promotion: {new_piece.type} created for team {new_piece.team}")
        update_display(WIN, grid, ROWS, WIDTH)
    else:
        print("Error: promotion not created")

def displayPromotionOptions(team):
    # Set the size of each promotion option icon
    icon_size = WIDTH // ROWS // 2
    # Set the starting position for the icons, this can be adjusted based on your UI layout
    start_x = WIDTH // 2 - 2 * icon_size
    start_y = 10  # Some margin from the top

    # Load the images for each promotion option
    queen_img = pygame.transform.scale(WQUEEN if team == 'W' else BQUEEN, (icon_size, icon_size))
    rook_img = pygame.transform.scale(WROOK if team == 'W' else BROOK, (icon_size, icon_size))
    bishop_img = pygame.transform.scale(WBISHOP if team == 'W' else BBISHOP, (icon_size, icon_size))
    knight_img = pygame.transform.scale(WKNIGHT if team == 'W' else BKNIGHT, (icon_size, icon_size))

    # Define rectangles for each option for click detection
    queen_rect = pygame.Rect(start_x, start_y, icon_size, icon_size)
    rook_rect = pygame.Rect(start_x + icon_size, start_y, icon_size, icon_size)
    bishop_rect = pygame.Rect(start_x + 2 * icon_size, start_y, icon_size, icon_size)
    knight_rect = pygame.Rect(start_x + 3 * icon_size, start_y, icon_size, icon_size)

    # Draw the options on the screen
    WIN.blit(queen_img, queen_rect)
    WIN.blit(rook_img, rook_rect)
    WIN.blit(bishop_img, bishop_rect)
    WIN.blit(knight_img, knight_rect)
    pygame.display.update()

    # Return the rectangles for click detection
    return {"queen": queen_rect, "rook": rook_rect, "bishop": bishop_rect, "knight": knight_rect}

def getPlayerChoice(promotion_options):
    # Loop until a valid choice is made
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get the mouse click position
                mouse_x, mouse_y = pygame.mouse.get_pos()

                # Check if the click is within any of the promotion option rectangles
                for piece, rect in promotion_options.items():
                    if rect.collidepoint(mouse_x, mouse_y):
                        return piece  # Return the name of the chosen piece

############ END PAWN PROMOTION ############

#checking for winner
def Chesscheck_win(grid):
    white_count = 0
    black_count = 0

    for row in grid:
        for node in row:
            if node.piece and node.piece.team == 'W':
                white_count += 1
            elif node.piece and node.piece.team == 'B':
                black_count += 1

    if white_count == 0:
        return 'B'  # Black wins
    elif black_count == 0:
        return 'W'  # White wins

    return None  # No winner yet
#win screen display
def display_Chesswin_screen(winner):
    font = pygame.font.Font(None, 65)

    # Create a black box
    grey_box_width = 800
    grey_box_height = 300
    grey_box = pygame.Surface((grey_box_width, grey_box_height))
    grey_box.fill((50, 50, 50))  # Black color

    # Get the dimensions of the text
    text = font.render(f"{winner} Wins! Press Esc for options", True, CHECKERSBLUE)
    text_width, text_height = text.get_size()
    HEIGHT = 600
    # Calculate the position of the black box
    box_x = (WIDTH - grey_box_width) // 2
    box_y = (HEIGHT - grey_box_height) // 2

    # Calculate the position of the text within the black box
    text_x = box_x + (grey_box_width - text_width) // 2
    text_y = box_y + (grey_box_height - text_height) // 2

    # Draw the black box
    WIN.blit(grey_box, (box_x, box_y))

    # Draw the text on top of the black box
    WIN.blit(text, (text_x, text_y))

    pygame.display.update()
    pygame.time.delay(3000)  # Display for 3 seconds

def make_testgrid(rows, width):
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
            if(i,j) in [(1,0), (1,1), (1,2), (2,3), (3,4), (1,5), (1,6), (1,7)]:
                node.piece = Piece('B','PAWN',BPAWN)
            if(i,j) in [(6,0), (6,1), (6,2), (6,3), (4,4), (6,5), (6,6), (6,7)]:
                node.piece = Piece('W','PAWN',WPAWN)

            #  Place rooks in the corners of the board.
            if i == 0 and j == 0 or i == 0 and j == 7:
                node.piece = Piece('B','ROOK', BROOK)
            if i == 7 and j == 0 or i == 7 and j == 5:
                node.piece = Piece('W','ROOK', WROOK)

            # Initialize knights next to the rooks.
            if i == 2 and j == 2 or i == 3 and j == 7:
                node.piece = Piece('B','KNIGHT', BKNIGHT)
            if i == 4 and j == 0 or i == 5 and j == 5:
                node.piece = Piece('W','KNIGHT', WKNIGHT)

            # Place bishops next to the knights.
            if i == 1 and j == 4 or i == 2 and j == 4:
                node.piece = Piece('B','BISHOP', BBISHOP)
            if i == 3 and j == 2 or i == 7 and j == 2:
                node.piece = Piece('W','BISHOP', WBISHOP)

            #  Set the king and queen in the middle of the first and last rows.
            if i == 0 and j == 4:
                node.piece = Piece('B','KING', BKING)
            if i == 7 and j == 6:
                node.piece = Piece('W','KING', WKING)
            if i == 2 and j == 7:
                node.piece = Piece('B','QUEEN', BQUEEN)
            if i == 7 and j == 3:
                node.piece = Piece('W','QUEEN', WQUEEN)

            # Add the node to the current row in the grid.
            grid[i].append(node)
    return grid

'''adding all of the checkers functions so both games are in one file'''
class checkersNode:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.x = int(row * width)
        self.y = int(col * width)
        self.colour = CHECKERSWHITE
        self.piece = None

    def draw(self, WIN):
        pygame.draw.rect(WIN, self.colour, (self.x, self.y, WIDTH / ROWS, WIDTH / ROWS))
        if self.piece:
            WIN.blit(self.piece.image, (self.x, self.y))


def update_checkersdisplay(win, grid, rows, width):
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_checkersgrid(win, rows, width)
    pygame.display.update()


def make_checkersgrid(rows, width):
    grid = []
    gap = width// rows
    count = 0
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = checkersNode(j,i, gap)
            if abs(i-j) % 2 == 0:
                node.colour=CHECKERSBLACK
            if (abs(i+j)%2==0) and (i<3):
                node.piece = checkersPiece('R')
            elif(abs(i+j)%2==0) and i>4:
                node.piece=checkersPiece('G')
            #count+=1
            grid[i].append(node)
    return grid


def draw_checkersgrid(win, rows, width):
    gap = width // ROWS
    for i in range(rows):
        pygame.draw.line(win, CHECKERSBLACK, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, CHECKERSBLACK, (j * gap, 0), (j * gap, width))


class checkersPiece:
    def __init__(self, team):
        self.team=team
        self.image= RED if self.team=='R' else GREEN
        self.type=None

    def draw(self, x, y):
        WIN.blit(self.image, (x,y))


def getCheckersNode(grid, rows, width):
    gap = width//rows
    RowX,RowY = pygame.mouse.get_pos()
    if RowX < 0 or RowX >= width or RowY < 0 or RowY >= width:
        return None  # Click is outside the board
    Row = RowX//gap
    Col = RowY//gap
    return (Col,Row)


def resetCheckersColours(grid, node):
    positions = generatePotentialCheckersMoves(node, grid)
    positions.append(node)

    for colouredNodes in positions:
        nodeX, nodeY = colouredNodes
        grid[nodeX][nodeY].colour = CHECKERSBLACK if abs(nodeX - nodeY) % 2 == 0 else CHECKERSWHITE

def HighlightpotentialCheckersMoves(piecePosition, grid):
    positions = generatePotentialCheckersMoves(piecePosition, grid)
    for position in positions:
        Column,Row = position
        grid[Column][Row].colour=CHECKERSBLUE

def checkersopposite(team):
    return "R" if team=="G" else "G"

def generatePotentialCheckersMoves(nodePosition, grid):
    checker = lambda x,y: x+y>=0 and x+y<8
    positions= []
    column, row = nodePosition
    if grid[column][row].piece:
        vectors = [[1, -1], [1, 1]] if grid[column][row].piece.team == "R" else [[-1, -1], [-1, 1]]
        if grid[column][row].piece.type=='KING':
            vectors = [[1, -1], [1, 1],[-1, -1], [-1, 1]]
        for vector in vectors:
            columnVector, rowVector = vector
            if checker(columnVector,column) and checker(rowVector,row):
                #grid[(column+columnVector)][(row+rowVector)].colour=ORANGE
                if not grid[(column+columnVector)][(row+rowVector)].piece:
                    positions.append((column + columnVector, row + rowVector))
                elif grid[column+columnVector][row+rowVector].piece and\
                        grid[column+columnVector][row+rowVector].piece.team==checkersopposite(grid[column][row].piece.team):

                    if checker((2* columnVector), column) and checker((2* rowVector), row) \
                            and not grid[(2* columnVector)+ column][(2* rowVector) + row].piece:
                        positions.append((2* columnVector+ column,2* rowVector+ row ))

    return positions

def checkershighlight(ClickedNode, Grid, OldHighlight):
    Column,Row = ClickedNode
    Grid[Column][Row].colour=CHECKERSORANGE
    if OldHighlight:
        resetCheckersColours(Grid, OldHighlight)
    HighlightpotentialCheckersMoves(ClickedNode, Grid)
    return (Column,Row)

def checkersmove(grid, piecePosition, newPosition):
    resetCheckersColours(grid, piecePosition)
    newColumn, newRow = newPosition
    oldColumn, oldRow = piecePosition

    piece = grid[oldColumn][oldRow].piece
    grid[newColumn][newRow].piece=piece
    grid[oldColumn][oldRow].piece = None

    if newColumn==7 and grid[newColumn][newRow].piece.team=='R':
        grid[newColumn][newRow].piece.type='KING'
        grid[newColumn][newRow].piece.image=REDKING
    if newColumn==0 and grid[newColumn][newRow].piece.team=='G':
        grid[newColumn][newRow].piece.type='KING'
        grid[newColumn][newRow].piece.image=GREENKING
    if abs(newColumn-oldColumn)==2 or abs(newRow-oldRow)==2:
        grid[int((newColumn+oldColumn)/2)][int((newRow+oldRow)/2)].piece = None
        return checkersopposite(grid[newColumn][newRow].piece.team)
    return checkersopposite(grid[newColumn][newRow].piece.team)

def checkersCheck_win(grid):
    red_count = 0
    green_count = 0

    for row in grid:
        for node in row:
            if node.piece and node.piece.team == 'R':
                red_count += 1
            elif node.piece and node.piece.team == 'G':
                green_count += 1

    if red_count == 0:
        return 'G'  # Green wins
    elif green_count == 0:
        return 'R'  # Red wins

    return None  # No winner yet

def display_Checkerswin_screen(winner):
    font = pygame.font.Font(None, 65)

    # Create a black box
    grey_box_width = 800
    grey_box_height = 300
    grey_box = pygame.Surface((grey_box_width, grey_box_height))
    grey_box.fill((50, 50, 50))  # Black color

    # Get the dimensions of the text
    text = font.render(f"{winner} Wins! Check terminal for options", True, CHECKERSBLUE)
    text_width, text_height = text.get_size()
    HEIGHT = 600
    # Calculate the position of the black box
    box_x = (WIDTH - grey_box_width) // 2
    box_y = (HEIGHT - grey_box_height) // 2

    # Calculate the position of the text within the black box
    text_x = box_x + (grey_box_width - text_width) // 2
    text_y = box_y + (grey_box_height - text_height) // 2

    # Draw the black box
    WIN.blit(grey_box, (box_x, box_y))

    # Draw the text on top of the black box
    WIN.blit(text, (text_x, text_y))

    pygame.display.update()
    pygame.time.delay(3000)  # Display for 3 seconds

def make_testcheckersgrid(rows, width):
    grid = []
    gap = width // rows

    for i in range(rows):
        grid.append([])
        for j in range(rows):
            node = checkersNode(j, i, gap)
            if abs(i - j) % 2 == 0:
                node.colour = CHECKERSBLACK

            if i == 0 and (j == 0 or j == 4):
                node.piece = checkersPiece('R')
            elif i == 1 and (j == 3 or j == 5 or j == 7):
                node.piece = checkersPiece('R')
            elif i == 2 and (j == 0 or j == 6):
                node.piece = checkersPiece('R')

            if i == 3 and j == 1:
                node.piece = checkersPiece('R')
            elif i == 3 and j == 3:
                node.piece = checkersPiece('G')

            if i == 5 and (j == 1 or j == 3 or j == 7):
                node.piece = checkersPiece('G')
            elif i == 6 and (j == 0 or j == 2 or j == 6):
                node.piece = checkersPiece('G')

            if i == 7 and j == 3:
                node.piece = checkersPiece('R')
                node.piece.type = 'KING'

            grid[i].append(node)

    return grid
'''end of checkers funtions'''

def draw_grid_labels(surface, top_left_x, top_left_y, cell_size):
    font = pygame.font.SysFont(None, 24)
    text_color = (255, 255, 255)  # White color for the text

    # Draw the column letters (A-H) on top and bottom
    for i in range(8):
        col_text = font.render(chr(65 + i), True, text_color)
        # Top
        surface.blit(col_text, (top_left_x + i * cell_size + cell_size // 2, top_left_y - 20))
        # Bottom
        surface.blit(col_text, (top_left_x + i * cell_size + cell_size // 2, top_left_y + 8 * cell_size))

    # Draw the row numbers (1-8) on left and right
    for i in range(8):
        row_text = font.render(str(8 - i), True, text_color)
        # Left
        surface.blit(row_text, (top_left_x - 20, top_left_y + i * cell_size + cell_size // 2))
        # Right
        surface.blit(row_text, (top_left_x + 8 * cell_size, top_left_y + i * cell_size + cell_size // 2))

def chessGame():
    # Track the currently highlighted piece, initially set to None
    highlightedPiece = None

    # Define the additional space needed for labels around the board
    label_space = 40  # Example space for labels
    # Increase the size of the window to accommodate the labels
    window_size = WIDTH + label_space * 2
    board_size = WIDTH
    cell_size = board_size // ROWS
    # Increase the grid size to accommodate labels
    background_size = board_size + 40  # Adding extra space for labels
    board_top_left_x = (background_size - board_size) // 2
    board_top_left_y = (background_size - board_size) // 2

    # Initialize the window with the new size
    WIN = pygame.display.set_mode((window_size, window_size))
    pause = False
    loop = True
    grid = make_grid(ROWS,WIDTH)
    currMove = 'W'
    lastMove = None

    #local functions for pause menu
    def restartChessGame():
        nonlocal loop, grid, currMove, lastMove, highlightedPiece
        loop = True
        grid = make_grid(ROWS, WIDTH)
        currMove = 'W'
        lastMove = None
        highlightedPiece = None

    def returnToMainMenu():
        nonlocal loop
        loop = False
        main()#going back to the main menu

    pause_buttons = None

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pause = not pause
                if pause:
                    pause_buttons = draw_pause(WIN)  # Update pause_buttons here

            #pause menu buttons, 1 to quit program, 2 to restart game, 3 to return to text menu
            if pause and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pause_buttons[0].collidepoint(mouse_x, mouse_y):
                    pygame.quit()
                    sys.exit()
                elif pause_buttons[1].collidepoint(mouse_x, mouse_y):
                    restartChessGame()
                    pause = False
                elif pause_buttons[2].collidepoint(mouse_x, mouse_y):
                    return returnToMainMenu()

            if event.type == pygame.MOUSEBUTTONDOWN and not pause:
                clickedNode = getNode(ROWS, WIDTH)
                if clickedNode is not None:  # Add this check
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
                                            update_display(WIN,grid,ROWS,WIDTH)
                                            winner = opposite(currMove)
                                            display_Chesswin_screen(winner)
                                        else:
                                            print("Stalemate!")
                                # Switch turns
                                currMove = opposite(currMove)
                            else:
                                # Move was not successful (would put/leave king in check), handle accordingly
                                print("Illegal move: would put/leave king in check.")
                                # Do not switch turns, allow the player to make another move
                    elif highlightedPiece == clickedNode:
                            pass
                    else:
                        if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                            if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                                highlightedPiece = highlight(clickedNode, grid, highlightedPiece, lastMove)

        if pause:
            current_screen = pygame.display.get_surface()
            draw_pause(current_screen)
            pygame.display.flip()  # Update the display
        else:
            # Clear the screen
            WIN.fill((0, 0, 0))  # Black background
            # Draw the grid labels
            draw_grid_labels(WIN, board_top_left_x, board_top_left_y, cell_size)
            update_display(WIN,grid,ROWS,WIDTH)
            #pygame.display.flip()
            winner = Chesscheck_win(grid)
            if winner:
                display_Chesswin_screen(winner)
                loop = False

def checkersGame():
# Track the currently highlighted piece, initially set to None
    highlightedPiece = None
    # Example: If WIDTH is the width of the game board, add additional space for UI elements if needed
    # Define the additional space needed for labels around the board
    label_space = 40  # Example space for labels
    # Increase the size of the window to accommodate the labels
    window_size = WIDTH + label_space * 2
    board_size = WIDTH
    cell_size = board_size // ROWS
    # Increase the grid size to accommodate labels
    background_size = board_size + 40  # Adding extra space for labels
    board_top_left_x = (background_size - board_size) // 2
    board_top_left_y = (background_size - board_size) // 2
    # Initialize the window with the new size
    WIN = pygame.display.set_mode((window_size, window_size))
    #pause option
    pause = False
    loop = True
    grid = make_checkersgrid(ROWS, WIDTH)
    currMove = 'G'

    def restartCheckersGame():
        nonlocal loop, grid, currMove, highlightedPiece
        loop = True
        grid = make_checkersgrid(ROWS, WIDTH)
        currMove = 'G'
        highlightedPiece = None

    def returnToMainMenu():
        nonlocal loop
        loop = False
        main()

    pause_buttons = None

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pause = not pause
                if pause:
                    pause_buttons = draw_pause(WIN)

            if pause and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pause_buttons[0].collidepoint(mouse_x, mouse_y):
                    pygame.quit()
                    sys.exit()
                elif pause_buttons[1].collidepoint(mouse_x, mouse_y):
                    restartCheckersGame()
                    pause = False
                elif pause_buttons[2].collidepoint(mouse_x, mouse_y):
                    return returnToMainMenu()

            if not pause and event.type == pygame.MOUSEBUTTONDOWN:
                clickedNode = getCheckersNode(grid,ROWS,WIDTH)
                if clickedNode is not None:
                    ClickedPositionColumn, ClickedPositionRow = clickedNode
                    if grid[ClickedPositionColumn][ClickedPositionRow].colour == CHECKERSBLUE:
                        if highlightedPiece:
                            pieceColumn, pieceRow = highlightedPiece
                            if currMove == grid[pieceColumn][pieceRow].piece.team:
                                resetCheckersColours(grid,highlightedPiece)
                                currMove=checkersmove(grid,highlightedPiece,clickedNode)
                    elif highlightedPiece == clickedNode:
                        pass
                    else:
                        if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                            if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                                highlightedPiece = checkershighlight(clickedNode, grid, highlightedPiece)
        if pause:
            current_screen = pygame.display.get_surface()
            draw_pause(current_screen)
            pygame.display.flip()
        else:
            # Clear the screen
            WIN.fill((0, 0, 0))  # Black background
             # Draw the grid labels for Checkers - ensure the coordinates are correct for board
            draw_grid_labels(WIN, board_top_left_x, board_top_left_y, cell_size)
            update_checkersdisplay(WIN, grid, ROWS, WIDTH)
            winner = checkersCheck_win(grid)
            if winner:
                display_Checkerswin_screen(winner)
                loop = False

def checkersDebug():
    highlightedPiece = None
    label_space = 40
    window_size = WIDTH + label_space * 2
    board_size = WIDTH
    cell_size = board_size // ROWS
    background_size = board_size + 40  # Adding extra space for labels
    board_top_left_x = (background_size - board_size) // 2
    board_top_left_y = (background_size - board_size) // 2
    WIN = pygame.display.set_mode((window_size, window_size))
    #pause option
    pause = False
    loop = True
    grid = make_testcheckersgrid(ROWS, WIDTH)
    currMove = 'G'

    def restartCheckersDebug():
        nonlocal loop, grid, currMove, highlightedPiece
        loop = True
        grid = make_testcheckersgrid(ROWS, WIDTH)
        currMove = 'G'
        highlightedPiece = None

    def returnToMainMenu():
        nonlocal loop
        loop = False
        main()

    pause_buttons = None

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pause = not pause
                if pause:
                    pause_buttons = draw_pause(WIN)

            if pause and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pause_buttons[0].collidepoint(mouse_x, mouse_y):
                    pygame.quit()
                    sys.exit()
                elif pause_buttons[1].collidepoint(mouse_x, mouse_y):
                    restartCheckersDebug()
                    pause = False
                elif pause_buttons[2].collidepoint(mouse_x, mouse_y):
                    return returnToMainMenu()

            if not pause and event.type == pygame.MOUSEBUTTONDOWN:
                clickedNode = getCheckersNode(grid,ROWS,WIDTH)
                if clickedNode is not None:
                    ClickedPositionColumn, ClickedPositionRow = clickedNode
                    if grid[ClickedPositionColumn][ClickedPositionRow].colour == CHECKERSBLUE:
                        if highlightedPiece:
                            pieceColumn, pieceRow = highlightedPiece
                            if currMove == grid[pieceColumn][pieceRow].piece.team:
                                resetCheckersColours(grid,highlightedPiece)
                                currMove=checkersmove(grid,highlightedPiece,clickedNode)
                    elif highlightedPiece == clickedNode:
                        pass
                    else:
                        if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                            if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                                highlightedPiece = checkershighlight(clickedNode, grid, highlightedPiece)
        if pause:
            current_screen = pygame.display.get_surface()
            draw_pause(current_screen)
            pygame.display.flip()
        else:
            # Clear the screen
            WIN.fill((0, 0, 0))  # Black background
             # Draw the grid labels for Checkers - ensure the coordinates are correct for board
            draw_grid_labels(WIN, board_top_left_x, board_top_left_y, cell_size)
            update_checkersdisplay(WIN,grid,ROWS,WIDTH)
            winner = checkersCheck_win(grid)
            if winner:
                display_Checkerswin_screen(winner)
                loop = False

def chessDebug():
    highlightedPiece = None
    label_space = 40
    window_size = WIDTH + label_space * 2
    board_size = WIDTH
    cell_size = board_size // ROWS
    # Increase the grid size to accommodate labels
    background_size = board_size + 40  # Adding extra space for labels
    board_top_left_x = (background_size - board_size) // 2
    board_top_left_y = (background_size - board_size) // 2
    WIN = pygame.display.set_mode((window_size, window_size))
    pause = False
    loop = True
    grid = make_testgrid(ROWS,WIDTH)
    currMove = 'W'
    lastMove = None
    checkMate = isKingInCheckmate(currMove, grid)

    def restartChessDebug():
        nonlocal loop, grid, currMove, lastMove, highlightedPiece
        loop = True
        grid = make_testgrid(ROWS, WIDTH)
        currMove = 'W'
        lastMove = None
        highlightedPiece = None

    def returnToMainMenu():
        nonlocal loop
        loop = False
        main(WIDTH, ROWS)#going back to the main menu

    pause_buttons = None

    while loop:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print('EXIT SUCCESSFUL')
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pause = not pause
                if pause:
                    pause_buttons = draw_pause(WIN)

            if pause and event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if pause_buttons[0].collidepoint(mouse_x, mouse_y):
                    pygame.quit()
                    sys.exit()
                elif pause_buttons[1].collidepoint(mouse_x, mouse_y):
                    restartChessDebug()
                    pause = False
                elif pause_buttons[2].collidepoint(mouse_x, mouse_y):
                    return returnToMainMenu()

            if not pause and event.type == pygame.MOUSEBUTTONDOWN:
                clickedNode = getNode(ROWS, WIDTH)
                if clickedNode is not None:
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
                                            #exit()
                                            winner = opposite(currMove)
                                            display_Chesswin_screen(winner)
                                        else:
                                            print("Stalemate!")
                                # Switch turns
                                currMove = opposite(currMove)
                            else:
                                # Move was not successful (would put/leave king in check), handle accordingly
                                print("Illegal move: would put/leave king in check.")
                                # Do not switch turns, allow the player to make another move
                    elif highlightedPiece == clickedNode:
                            pass
                    else:
                        if grid[ClickedPositionColumn][ClickedPositionRow].piece:
                            if currMove == grid[ClickedPositionColumn][ClickedPositionRow].piece.team:
                                highlightedPiece = highlight(clickedNode, grid, highlightedPiece, lastMove)
            if pause:
                current_screen = pygame.display.get_surface()
                draw_pause(current_screen)
                pygame.display.flip()
            else:
                WIN.fill((0, 0, 0))
                draw_grid_labels(WIN, board_top_left_x, board_top_left_y, cell_size)
                update_display(WIN,grid,ROWS,WIDTH)
                winner = Chesscheck_win(grid)
                if winner:
                    display_Chesswin_screen(winner)
                    loop = False

''' new main function to work with both games'''
def main():
    print("Menu")
    print("Choose one of the following options")
    print("Option1: Play Checkers")
    print("Option2: Play Chess ")
    print("Option3: Checkers Test")
    print("Option4: Chess Test")
    while True:
        choice = input("Enter your choice 1, 2, 3, 4, or q to exit")

        if choice == '1':
            checkersGame()

        elif choice == '2':
            chessGame()

        elif choice == '3':
            checkersDebug()

        elif choice == '4':
            chessDebug()

        elif choice == 'q':
            pygame.quit()
            sys.exit()

        else:
            print("invalid option please enter 1 2 3 4 or q: ")
main()
