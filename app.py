import dash
from dash import Dash, DiskcacheManager, html, dcc, ctx
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.express as px

import diskcache
import dash_auth
from decouple import config

import numpy as np

import onnxruntime

import requests
from io import BytesIO
from PIL import Image

from time import sleep


### APP CONFIG ###

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

app = Dash(
    __name__, background_callback_manager=background_callback_manager,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Silomap', update_title=None
)
server = app.server

auth = dash_auth.BasicAuth(
    app,
    {config('SILOMAP_USER'): config('SILOMAP_PASSWORD')}
)

mapbox_token = config('MAPBOX_TOKEN')
mapbox_style = 'mapbox://styles/mapbox/satellite-v9'

path_logo = 'logo.png'
path_model = './models/model.onnx'

### LAYOUT ###

init_coords = dict(
    lat=49.47557512435642,
    lon=0.23256826217428248
)
init_zoom = 15

map_config = dict(
    autosizable=True,
    modeBarButtons = [
        ['toImage'],
        ['pan2d', 'select2d'],
        ['resetViewMapbox']
    ]
)

image_config = dict(
    autosizable=True,
    modeBarButtons = [
        ['toImage'],
        ['pan2d'],
        ['zoom2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d']
    ]
)

def init_map():
    fig = go.Figure()
    fig.update_layout(
        autosize=True,
        margin=dict(t=0, r=0, b=0, l=0),
        mapbox=dict(
            accesstoken = mapbox_token,
            style = mapbox_style,
            center=init_coords,
            zoom=init_zoom,
        )
    )
    fig = fig.add_scattermapbox()
    return fig

markdown_instructions = dcc.Markdown("""
            ## Instructions:\n
            1. Choose a **zoom** level between 16 and 19 for the tiles\n
            2. Select an area on the map and press the **select** button\n
            3. Press the **load** button (be careful not to load to much tiles!)\n
            4. Press the **process** button to run the silo detection\n
            """)

modal_confirm = dbc.Modal(
    id='modal-confirm',
    keyboard=False,
    backdrop='static',
    centered=True,
    children=[
        dbc.ModalHeader(
            dbc.ModalTitle('Load confirmation'), close_button=False
        ),
        dbc.ModalBody([
            html.Div(id='modal-body'),
            dbc.Alert(
                id='modal-alert',
                children='You cannot load more than 50 tiles at once.',
                dismissable=False,
                color='danger',
                is_open=False
            )
        ]),
        dbc.ModalFooter([
            dbc.Button(id='modal-cancel', children='cancel', outline=True, color='danger'),
            dbc.Button(id='modal-proceed', children='proceed', outline=True, color='success'),
        ])
    ]
)

tooltip_selected = dbc.Tooltip(
    id='tooltip-selected',
    is_open=False,
    trigger=None,
    target='button-select',
    placement='left'
)

app.layout = html.Div([

    dcc.Store(id='store-select'),
    dcc.Download(id='image-downloader'),


    dbc.Row([
        dbc.Col(html.Img(src=dash.get_asset_url(path_logo), width=230, height=230), width=2),
        dbc.Col(
            markdown_instructions,
            width=4,
            align='center',
        ),
        dbc.Col([
            dbc.Row([
                dbc.Col(dcc.Markdown('### Zoom:'), style={'float': 'left'}),
                dbc.Col(dbc.Select(id='select-zoom', options=['16', '17', '18', '19'], value='17', style={'width': '80px'}))
            ]),
            dbc.Row([
                dbc.Col(dcc.Markdown('### Segmentaion:'), style={'float': 'left'}),
                dbc.Col(dbc.Switch(id='switch-segm', input_style={'width': '3rem', 'height': '2rem'}))
            ]),
            html.Br(),
            dbc.ButtonGroup([
                dbc.Button(id='button-select', children='select', outline=True, color='success'),
                dbc.Button(id='button-load', children='load', outline=True, color='success'),
                dbc.Button(id='button-process', children='process', outline=True, color='success'),
            ], size='lg', style={'padding-bottom': '10px'}),
            dbc.Fade(
                dbc.Progress(id='progress', striped=True, animated=True, color="success", className="mb-3", style={'width': '265px'}),
                id='fade-progress',
                is_in=False
            ),
            modal_confirm,
            tooltip_selected,
        ], width=3, align='center'),
        dbc.Col([
            dbc.ButtonGroup([
                dbc.Button(id='button-save-image', children='save image', outline=True, color='success'),
                dbc.Button(id='button-save-pred', children='save prediction', outline=True, color='success')
            ], size='lg')
        ], width=3, align='center'),
    ]),

    dbc.Row([
        dbc.Col(
            dcc.Graph(id='graph-map', figure=init_map(), config=map_config, style={'width': '49vw', 'height': '75vh'})
        ),
        dbc.Col(
            dbc.Spinner([
                    dcc.Graph(id='graph-image', config=image_config, style={'width': '49vw', 'height': '75vh'}),
                    dcc.Store(id='store-image'),
                    dcc.Store(id='store-pred')
                ],
                color='success', spinner_style={'width': '3rem', 'height': '3rem'}
            )
            
        )
    ])
], style={'overflow': 'hidden', 'background': '#E5F6DF'})

### UTILS ###

def tile_numbers_from_point(lat_deg, lon_deg, zoom):
    lat_rad = np.radians(lat_deg)
    n = 2 ** zoom
    x = int((lon_deg + 180.0) / 360.0 * n)
    y = int((1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) / 2.0 * n)
    return x, y

def mapbox_url(x, y, zoom):
    return f'https://api.mapbox.com/v4/mapbox.satellite/{zoom}/{x}/{y}.jpg90?access_token={mapbox_token}'

def get_image_figure(image, empty=False):
    if empty:
        fig = go.Figure()
    else:
        fig = px.imshow(image)

    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor='x')
    fig.update_layout(
        margin=dict(t=0, r=0, b=0, l=0)
    )
    return fig

def get_segm_image(pred, image):
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    segm = np.stack((gray, gray, gray), -1).astype(np.uint8)

    segm[pred != 0] = segm[pred != 0] * [0.8, 0.3, 0]
    return segm


### CALLBACKS ###

@app.callback(
    Output('store-select', 'data'),
    Input('button-select', 'n_clicks'),
    [State('graph-map', 'selectedData'), State('select-zoom', 'value')]
)
def on_button_select(n_clicks, selectedData, zoom):
    if selectedData is None:
        return None

    coords = selectedData['range']['mapbox']
    lon0, lat0 = coords[0]
    lon1, lat1 = coords[1]

    x0, y0 = tile_numbers_from_point(lat0, lon0, int(zoom))
    x1, y1 = tile_numbers_from_point(lat1, lon1, int(zoom))

    n_tiles = (x1-x0+1) * (y1-y0+1)

    return {'coords': coords, 'tiles': [x0, y0, x1, y1], 'n_tiles': n_tiles, 'zoom': zoom}
    
@app.callback(
    [Output('tooltip-selected', 'is_open'), Output('tooltip-selected', 'children')],
    Input('store-select', 'data')
)
def toggle_selected_tooltip(data):
    if data is None:
        return False, ''
    
    return True, f'{data["n_tiles"]} tiles selected'

@app.callback(
    [
        Output('button-load', 'disabled'), Output('button-process', 'disabled'),
        Output('button-save-image', 'disabled'), Output('button-save-pred', 'disabled'),
    ],
    [Input('store-select', 'data'), Input('store-image', 'data'), Input('store-pred', 'data')]
)
def toggle_disable_buttons(data, image, pred):
    disable_load = data is None
    disable_process = image is None
    disable_save_image = disable_process
    disable_save_pred = pred is None
    return disable_load, disable_process, disable_save_image, disable_save_pred
    

@app.callback(
    [
        Output('modal-body', 'children'), Output('modal-alert', 'is_open'),
        Output('modal-proceed', 'disabled')
    ],
    Input('store-select', 'data')
)
def set_modal(data):
    if data is None:
        return '', False, False

    body_info = f'You are about to load {data["n_tiles"]} tiles at zoom level {data["zoom"]}.'
    if data['n_tiles'] > 50:
        return body_info, True, True
    return body_info, False, False

    

@app.callback(
    Output('modal-confirm', 'is_open'),
    [Input('button-load', 'n_clicks'), Input('modal-proceed', 'n_clicks'), Input('modal-cancel', 'n_clicks')],
)
def toggle_load_confirm(n_open, n_proceed, n_cancel):
    if ctx.triggered_id == 'button-load':
        return True
    
    if ctx.triggered_id == 'modal-proceed':
        return False
    
    if ctx.triggered_id == 'modal-cancel':
        return False
    return False

@app.callback(
    Output('store-image', 'data'),
    Input('modal-proceed', 'n_clicks'),
    State('store-select', 'data'),
    background=True,
    running=[
        (Output('modal-proceed', 'disabled'), True, False),
        (Output('modal-proceed', 'children'), [dbc.Spinner(size='sm'), 'proceed'], 'proceed'),
        (Output('fade-progress', 'is_in'), True, False)
    ],
    progress=[Output('progress', 'value'), Output('progress', 'max')],
    progress_default=(0, 100)
)
def load_and_stitch_tiles(set_progress, n_clicks, data):
    if data is None:
        return None
    
    x0, y0, x1, y1 = data['tiles']
    dx = x1 - x0 + 1
    dy = y1 - y0 + 1

    c = 0

    image = np.zeros((256 * dy, 256 * dx, 3))
    for x in range(x0, x1+1):
        for y in range(y0, y1+1):
            # TODO: handle bad responses
            r = requests.get(mapbox_url(x, y, data['zoom']))
            img = Image.open(BytesIO(r.content))
            img = np.array(img)

            i = x - x0
            j = y - y0

            image[j*256:(j+1)*256, i*256:(i+1)*256] = img

            c += 1
            set_progress((c, dx * dy))
            sleep(0.1)

    return image


@app.callback(
    Output('store-pred', 'data'),
    Input('button-process', 'n_clicks'),
    [State('store-image', 'data'), State('switch-segm', 'disabled'), State('store-pred', 'data')]
)
def calc_pred(n_clicks, image, switch_disabled, pred):
    if image is None:
        return None

    if not switch_disabled:
        return pred
    
    ort_session = onnxruntime.InferenceSession(path_model)

    full_input = np.array(image).astype(np.float32) / 255.
    width, height, _ = full_input.shape
    
    full_pred = np.zeros((width, height))

    for i in range(width // 256):
        for j in range(height // 256):
            input = full_input[i*256:(i+1)*256, j*256:(j+1)*256,:]
            input = np.moveaxis(input, -1, 0)
            input = input.reshape((1, 3, 256, 256))
            pred = ort_session.run(None, {'image': input})
            pred = pred[0].reshape((256, 256))
            pred = np.rint(pred)
            full_pred[i*256:(i+1)*256, j*256:(j+1)*256] = pred
    
    return full_pred

@app.callback(
    [Output('switch-segm', 'disabled'), Output('switch-segm', 'value')],
    [Input('modal-proceed', 'n_clicks'), Input('store-pred', 'data')]
)
def toggle_segm_switch(n_click, pred):
    if pred is None or ctx.triggered_id == 'modal-proceed':
        return True, False
    return False, True


@app.callback(
    Output('graph-image', 'figure'),
    [Input('store-image', 'data'), Input('switch-segm', 'value')],
    State('store-pred', 'data')
)
def update_image(image, display_segm, pred):
    if image is None:
        return get_image_figure(None, empty=True)
    
    image = np.array(image).astype(np.uint8)

    if ctx.triggered_id == 'switch-segm' and display_segm:
        pred = np.array(pred)
        segm_image = get_segm_image(pred, image)
        return get_image_figure(segm_image)

    return get_image_figure(image)


@app.callback(
    Output('image-downloader', 'data'),
    [Input('button-save-image', 'n_clicks'), Input('button-save-pred', 'n_clicks')],
    [State('store-image', 'data'), State('store-pred', 'data')],
    prevent_initial_call=True,
    background=True,
    running = [
        (Output('button-save-image', 'disabled'), True, False),
        (Output('button-save-pred', 'disabled'), True, False),
    ]
)
def save_images(n_clicks0, n_clicks1, image, pred):

    if ctx.triggered_id == 'button-save-image':
        image = Image.fromarray(np.array(image).astype(np.uint8))
        return dcc.send_bytes(lambda bytes_io: image.save(bytes_io, format='PNG'), 'image.png')

    if ctx.triggered_id == 'button-save-pred':
        pred = Image.fromarray(np.array(pred).astype(np.uint8) * 255, mode='L')
        return dcc.send_bytes(lambda bytes_io: pred.save(bytes_io, format='PNG'), 'prediction.png')

    return dash.no_update
        
if __name__ == '__main__':
    app.run()