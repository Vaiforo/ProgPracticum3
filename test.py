import PySimpleGUI as sg

layout = [
    [sg.Text('Time:'), sg.Slider(range=(0, 24), orientation='h',
                                 key='-TIME-', enable_events=True)],
    [sg.Graph(canvas_size=(600, 400), graph_bottom_left=(
        0, 0), graph_top_right=(100, 100), key='-MAP-')],
    [sg.Multiline(size=(80, 5), key='-STATS-', disabled=True)]
]
window = sg.Window('Ecosystem Simulator', layout, finalize=True)
