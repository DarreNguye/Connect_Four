from flask import Flask, jsonify, request, send_from_directory
from bots import BOTS
from engine import Board

app = Flask(__name__, static_folder=".")

game = Board()

def state(last_move=None, bot_move=None):
    line = game.winning_line()
    return jsonify(
        grid=game.grid,
        side=game.side,
        winner=game.winner(),
        legal=game.legal_moves(),
        history=game.history,
        winning_line=line,
        last_move=last_move,
        bot_move=bot_move,
        bots=sorted(BOTS),
    )


@app.get("/")
def index():
    return send_from_directory(".", "index.html")


@app.post("/api/new")
def new_game():
    global game
    game = Board()
    return state()


@app.post("/api/move")
def move():
    col = int(request.json["col"])
    if game.is_over():
        return jsonify(error="game is over"), 400
    if col not in game.legal_moves():
        return jsonify(error=f"column {col} is full"), 400
    game.play(col)
    return state(last_move=col)


@app.post("/api/undo")
def undo():
    if game.history:
        game.undo()
    return state()


@app.post("/api/bot")
def bot_move():
    if game.is_over():
        return jsonify(error="game is over"), 400
    name = request.json.get("bot", "tactical")
    if name not in BOTS:
        return jsonify(error=f"unknown bot {name!r}"), 400
    col = BOTS[name](game)
    if col not in game.legal_moves():
        return jsonify(error=f"bot {name!r} chose illegal column {col}"), 500
    game.play(col)
    return state(last_move=col, bot_move=col)


@app.get("/api/state")
def get_state():
    return state()


if __name__ == "__main__":
    app.run(debug=True, port=5001)