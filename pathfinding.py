import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, RadioButtons
from collections import deque
import heapq
import time


ROWS, COLS = 18, 22

WALL = 1
FREE = 0
START_R, START_C = 2, 2
END_R, END_C = 15, 19

raw_walls = [
    (1,5),(2,5),(3,5),(4,5),(4,6),(4,7),(4,8),
    (1,10),(2,10),(3,10),(3,11),(3,12),(3,13),
    (6,1),(7,1),(8,1),(9,1),(9,2),(9,3),(9,4),
    (6,5),(6,6),(6,7),(7,7),(8,7),(8,8),(8,9),(8,10),
    (11,3),(12,3),(13,3),(14,3),(14,4),(14,5),(14,6),
    (11,7),(11,8),(11,9),(11,10),(12,10),(13,10),
    (6,13),(7,13),(8,13),(9,13),(9,14),(9,15),(9,16),
    (6,17),(7,17),(8,17),(8,16),(7,16),
    (12,14),(12,15),(12,16),(13,16),(14,16),(14,15),(14,14),
    (5,19),(6,19),(7,19),(7,20),(7,21),
    (11,18),(12,18),(12,19),(12,20),
    (3,16),(3,17),(3,18),(4,18),(5,18),
    (15,8),(15,9),(15,10),(16,10),(16,11),
    (1,14),(1,15),(1,16),(2,16),(2,17),
]

MOVES = [(-1,0),(0,1),(1,0),(1,1),(0,-1),(-1,-1)]

PALETTE = {
    "bg"       : "#1a1a2e",
    "wall"     : "#16213e",
    "free"     : "#0f3460",
    "frontier" : "#f5a623",
    "explored" : "#4a90d9",
    "path"     : "#e94560",
    "start"    : "#00d4aa",
    "end"      : "#ff6b9d",
    "text"     : "#ffffff",
    "grid_line": "#1e2a4a",
}

algo_names = ["BFS", "DFS", "UCS", "DLS", "IDDFS", "Bidirectional"]

state = {
    "grid"        : None,
    "current_algo": 0,
    "running"     : False,
    "step_gen"    : None,
    "delay"       : 0.03,
    "stats"       : {"explored": 0, "path_len": 0, "time": 0.0},
    "start_time"  : 0,
}


def build_grid():
    g = np.zeros((ROWS, COLS), dtype=int)
    for r, c in raw_walls:
        if 0 <= r < ROWS and 0 <= c < COLS:
            g[r][c] = WALL
    return g


def neighbors(grid, r, c):
    result = []
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and grid[nr][nc] == FREE:
            result.append((nr, nc))
    return result


def reconstruct(came_from, node):
    path = []
    while node is not None:
        path.append(node)
        node = came_from.get(node)
    return list(reversed(path))


def bfs_gen(grid):
    queue = deque()
    queue.append((START_R, START_C))
    visited = {(START_R, START_C)}
    came_from = {(START_R, START_C): None}
    while queue:
        r, c = queue.popleft()
        yield "explore", (r, c), None
        if (r, c) == (END_R, END_C):
            yield "done", None, reconstruct(came_from, (END_R, END_C))
            return
        for nr, nc in neighbors(grid, r, c):
            if (nr, nc) not in visited:
                visited.add((nr, nc))
                came_from[(nr, nc)] = (r, c)
                queue.append((nr, nc))
                yield "frontier", (nr, nc), None
    yield "done", None, []


def dfs_gen(grid):
    stack = [(START_R, START_C)]
    visited = set()
    came_from = {(START_R, START_C): None}
    while stack:
        r, c = stack.pop()
        if (r, c) in visited:
            continue
        visited.add((r, c))
        yield "explore", (r, c), None
        if (r, c) == (END_R, END_C):
            yield "done", None, reconstruct(came_from, (END_R, END_C))
            return
        for nr, nc in reversed(neighbors(grid, r, c)):
            if (nr, nc) not in visited:
                if (nr, nc) not in came_from:
                    came_from[(nr, nc)] = (r, c)
                stack.append((nr, nc))
                yield "frontier", (nr, nc), None
    yield "done", None, []


def ucs_gen(grid):
    heap = [(0, START_R, START_C)]
    cost_so_far = {(START_R, START_C): 0}
    came_from = {(START_R, START_C): None}
    explored = set()
    while heap:
        cost, r, c = heapq.heappop(heap)
        if (r, c) in explored:
            continue
        explored.add((r, c))
        yield "explore", (r, c), None
        if (r, c) == (END_R, END_C):
            yield "done", None, reconstruct(came_from, (END_R, END_C))
            return
        for nr, nc in neighbors(grid, r, c):
            step = 1.414 if (nr - r != 0 and nc - c != 0) else 1.0
            new_cost = cost + step
            if (nr, nc) not in cost_so_far or new_cost < cost_so_far[(nr, nc)]:
                cost_so_far[(nr, nc)] = new_cost
                came_from[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, nr, nc))
                yield "frontier", (nr, nc), None
    yield "done", None, []


def dls_inner(grid, node, goal, limit, visited, came_from):
    if node == goal:
        return True
    if limit == 0:
        return False
    visited.add(node)
    r, c = node
    for nr, nc in neighbors(grid, r, c):
        if (nr, nc) not in visited:
            came_from[(nr, nc)] = node
            found = dls_inner(grid, (nr, nc), goal, limit - 1, visited, came_from)
            if found:
                return True
            visited.discard((nr, nc))
    return False


def dls_gen(grid, depth_limit=12):
    came_from = {(START_R, START_C): None}
    visited = {(START_R, START_C)}
    yield "explore", (START_R, START_C), None

    found = dls_inner(grid, (START_R, START_C), (END_R, END_C), depth_limit, visited, came_from)

    for node in visited:
        if node != (START_R, START_C):
            yield "explore", node, None

    if found:
        yield "done", None, reconstruct(came_from, (END_R, END_C))
    else:
        yield "done", None, []


def iddfs_gen(grid):
    for depth in range(1, ROWS * COLS):
        came_from = {(START_R, START_C): None}
        visited = {(START_R, START_C)}
        found = dls_inner(grid, (START_R, START_C), (END_R, END_C), depth, visited, came_from)
        for node in visited:
            yield "explore", node, None
        if found:
            yield "done", None, reconstruct(came_from, (END_R, END_C))
            return
    yield "done", None, []


def bidir_gen(grid):
    fwd_queue = deque([(START_R, START_C)])
    bwd_queue = deque([(END_R, END_C)])
    fwd_visited = {(START_R, START_C): None}
    bwd_visited = {(END_R, END_C): None}

    def build_bidir_path(meet):
        path_fwd = reconstruct(fwd_visited, meet)
        path_bwd = []
        node = bwd_visited.get(meet)
        while node is not None:
            path_bwd.append(node)
            node = bwd_visited.get(node)
        return path_fwd + path_bwd

    while fwd_queue or bwd_queue:
        if fwd_queue:
            r, c = fwd_queue.popleft()
            yield "explore", (r, c), None
            for nr, nc in neighbors(grid, r, c):
                if (nr, nc) not in fwd_visited:
                    fwd_visited[(nr, nc)] = (r, c)
                    fwd_queue.append((nr, nc))
                    yield "frontier", (nr, nc), None
                if (nr, nc) in bwd_visited:
                    yield "done", None, build_bidir_path((nr, nc))
                    return

        if bwd_queue:
            r, c = bwd_queue.popleft()
            yield "explore", (r, c), None
            for nr, nc in neighbors(grid, r, c):
                if (nr, nc) not in bwd_visited:
                    bwd_visited[(nr, nc)] = (r, c)
                    bwd_queue.append((nr, nc))
                    yield "frontier", (nr, nc), None
                if (nr, nc) in fwd_visited:
                    yield "done", None, build_bidir_path((nr, nc))
                    return

    yield "done", None, []


generators = [bfs_gen, dfs_gen, ucs_gen, dls_gen, iddfs_gen, bidir_gen]

cell_colors = None
ax_grid = None
ax_info = None
fig = None
img_obj = None


def make_color_array(grid):
    color_map = np.zeros((ROWS, COLS, 3))

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

    wall_rgb  = hex_to_rgb(PALETTE["wall"])
    free_rgb  = hex_to_rgb(PALETTE["free"])
    start_rgb = hex_to_rgb(PALETTE["start"])
    end_rgb   = hex_to_rgb(PALETTE["end"])

    for r in range(ROWS):
        for c in range(COLS):
            if grid[r][c] == WALL:
                color_map[r, c] = wall_rgb
            else:
                color_map[r, c] = free_rgb

    color_map[START_R, START_C] = start_rgb
    color_map[END_R,   END_C  ] = end_rgb
    return color_map


def reset_visualization():
    global cell_colors, img_obj
    grid = state["grid"]
    cell_colors = make_color_array(grid)
    img_obj.set_data(cell_colors)
    state["stats"] = {"explored": 0, "path_len": 0, "time": 0.0}
    update_info_panel()
    fig.canvas.draw_idle()


def update_info_panel():
    ax_info.clear()
    ax_info.set_facecolor(PALETTE["bg"])
    ax_info.axis("off")

    name = algo_names[state["current_algo"]]
    s = state["stats"]

    descriptions = {
        "BFS"         : "Explores level by level.\nGuarantees shortest path.",
        "DFS"         : "Dives deep before backtracking.\nNot guaranteed optimal.",
        "UCS"         : "Expands cheapest cost first.\nOptimal with variable costs.",
        "DLS"         : "DFS with depth limit=12.\nMay miss deep targets.",
        "IDDFS"       : "Repeated DFS with growing\nlimits. Optimal + memory safe.",
        "Bidirectional": "Searches from both ends.\nMeets in the middle.",
    }

    legend_items = [
        mpatches.Patch(color=PALETTE["start"],    label="Start"),
        mpatches.Patch(color=PALETTE["end"],      label="Target"),
        mpatches.Patch(color=PALETTE["frontier"], label="Frontier"),
        mpatches.Patch(color=PALETTE["explored"], label="Explored"),
        mpatches.Patch(color=PALETTE["path"],     label="Final Path"),
        mpatches.Patch(color=PALETTE["wall"],     label="Wall"),
    ]

    ax_info.legend(handles=legend_items, loc="upper left",
                   facecolor="#0f3460", edgecolor="#4a90d9",
                   labelcolor=PALETTE["text"], fontsize=9, framealpha=0.9)

    info_text = (
        f"Algorithm:  {name}\n\n"
        f"{descriptions.get(name, '')}\n\n"
        f"Nodes Explored: {s['explored']}\n"
        f"Path Length:    {s['path_len']}\n"
        f"Time Elapsed:   {s['time']:.2f}s\n\n"
        f"Movement Order:\n"
        f"Up → Right → Down\n"
        f"Down-Right → Left\n"
        f"Top-Left (diagonals)"
    )

    ax_info.text(0.05, 0.88, info_text, transform=ax_info.transAxes,
                 color=PALETTE["text"], fontsize=9, va="top",
                 fontfamily="monospace", linespacing=1.6)


def hex_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]


def run_step(event=None):
    global cell_colors

    if not state["running"]:
        return

    gen = state["step_gen"]
    if gen is None:
        return

    try:
        kind, node, path = next(gen)

        if kind == "frontier" and node:
            r, c = node
            if (r, c) not in [(START_R, START_C), (END_R, END_C)]:
                cell_colors[r, c] = hex_rgb(PALETTE["frontier"])
                state["stats"]["explored"] += 1

        elif kind == "explore" and node:
            r, c = node
            if (r, c) not in [(START_R, START_C), (END_R, END_C)]:
                cell_colors[r, c] = hex_rgb(PALETTE["explored"])
                state["stats"]["explored"] += 1

        elif kind == "done":
            state["running"] = False
            state["stats"]["time"] = time.time() - state["start_time"]
            if path:
                for r, c in path:
                    if (r, c) not in [(START_R, START_C), (END_R, END_C)]:
                        cell_colors[r, c] = hex_rgb(PALETTE["path"])
                state["stats"]["path_len"] = len(path)
            img_obj.set_data(cell_colors)
            update_info_panel()
            fig.canvas.draw_idle()
            return

        state["stats"]["time"] = time.time() - state["start_time"]
        img_obj.set_data(cell_colors)
        update_info_panel()
        fig.canvas.draw_idle()
        plt.pause(state["delay"])

        if state["running"]:
            fig.canvas.mpl_connect("draw_event", lambda e: None)
            run_step()

    except StopIteration:
        state["running"] = False


def on_run(event):
    reset_visualization()
    idx = state["current_algo"]
    gen_fn = generators[idx]
    if idx == 3:
        state["step_gen"] = gen_fn(state["grid"], depth_limit=12)
    else:
        state["step_gen"] = gen_fn(state["grid"])
    state["running"] = True
    state["start_time"] = time.time()
    run_step()


def on_reset(event):
    state["running"] = False
    state["step_gen"] = None
    reset_visualization()


def on_algo_change(label):
    state["running"] = False
    state["step_gen"] = None
    state["current_algo"] = algo_names.index(label)
    reset_visualization()


def on_speed(event):
    speeds = [0.08, 0.03, 0.01, 0.001]
    current = state["delay"]
    idx = speeds.index(current) if current in speeds else 1
    state["delay"] = speeds[(idx + 1) % len(speeds)]
    speed_labels = {0.08: "Slow", 0.03: "Normal", 0.01: "Fast", 0.001: "Turbo"}
    btn_speed.label.set_text(f"Speed: {speed_labels[state['delay']]}")
    fig.canvas.draw_idle()


def main():
    global cell_colors, ax_grid, ax_info, fig, img_obj
    global btn_run, btn_reset, btn_speed

    state["grid"] = build_grid()
    cell_colors = make_color_array(state["grid"])

    fig = plt.figure(figsize=(16, 9), facecolor=PALETTE["bg"])
    fig.canvas.manager.set_window_title("AI Pathfinder — Uninformed Search Visualizer")

    ax_grid = fig.add_axes([0.01, 0.12, 0.62, 0.86])
    ax_grid.set_facecolor(PALETTE["bg"])
    ax_grid.set_title("Grid World", color=PALETTE["text"], fontsize=13, pad=8)
    ax_grid.tick_params(colors=PALETTE["text"])
    for spine in ax_grid.spines.values():
        spine.set_edgecolor("#4a90d9")

    img_obj = ax_grid.imshow(cell_colors, interpolation="nearest", aspect="equal")

    ax_grid.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax_grid.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)
    ax_grid.grid(which="minor", color=PALETTE["grid_line"], linewidth=0.5)
    ax_grid.tick_params(which="minor", size=0)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    ax_info = fig.add_axes([0.65, 0.20, 0.33, 0.78])
    ax_info.set_facecolor(PALETTE["bg"])

    ax_radio = fig.add_axes([0.65, 0.01, 0.18, 0.18])
    ax_radio.set_facecolor(PALETTE["bg"])
    radio = RadioButtons(ax_radio, algo_names, activecolor=PALETTE["frontier"])
    for label in radio.labels:
        label.set_color(PALETTE["text"])
        label.set_fontsize(8)
    radio.on_clicked(on_algo_change)

    ax_btn_run   = fig.add_axes([0.85, 0.08, 0.07, 0.05])
    ax_btn_reset = fig.add_axes([0.85, 0.02, 0.07, 0.05])
    ax_btn_speed = fig.add_axes([0.85, 0.14, 0.07, 0.05])

    btn_run   = Button(ax_btn_run,   "Run",          color="#0f3460", hovercolor="#16213e")
    btn_reset = Button(ax_btn_reset, "Reset",        color="#0f3460", hovercolor="#16213e")
    btn_speed = Button(ax_btn_speed, "Speed: Normal",color="#0f3460", hovercolor="#16213e")

    for btn in [btn_run, btn_reset, btn_speed]:
        btn.label.set_color(PALETTE["text"])
        btn.label.set_fontsize(8)

    btn_run.on_clicked(on_run)
    btn_reset.on_clicked(on_reset)
    btn_speed.on_clicked(on_speed)

    update_info_panel()

    fig.text(0.01, 0.02,
             "Select algorithm Press Run to visualize  |  Reset clears the grid",
             color="#4a90d9", fontsize=8)

    plt.show()


main()