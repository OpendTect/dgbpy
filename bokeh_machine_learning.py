#__________________________________________________________________________
#
# (C) dGB Beheer B.V.; (LICENSE) http://opendtect.org/OpendTect_license.txt
# Author:        A. Huck
# Date:          Jan 2019
#
# _________________________________________________________________________


from random import random

from bokeh.layouts import row, column
from bokeh.models import Button, CustomJS, ColumnDataSource
from bokeh.models.widgets import Panel, Tabs
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

from io import StringIO
import base64


# create a plot and style its properties
p1 = figure(x_range=(0, 100), y_range=(0, 100), toolbar_location=None)
p1.border_fill_color = 'black'
p1.background_fill_color = 'black'
p1.outline_line_color = None
p1.grid.grid_line_color = None

# add a text renderer to our plot (no data yet)
r = p1.text(x=[], y=[], text=[], text_color=[], text_font_size="20pt",
           text_baseline="middle", text_align="center")

i = 0

ds = r.data_source

# create a callback that will add a number in a random location
def callback():
    global i

    # BEST PRACTICE --- update .data in one step with a new dict
    new_data = dict()
    new_data['x'] = ds.data['x'] + [random()*70 + 15]
    new_data['y'] = ds.data['y'] + [random()*70 + 15]
    new_data['text_color'] = ds.data['text_color'] + [RdYlBu3[i%3]]
    new_data['text'] = ds.data['text'] + [str(i)]
    ds.data = new_data

    i = i + 1

# add a button widget and configure with the call back
button = Button(label="Press Me")
button.on_click(callback)

# put the button and plot in a layout and add to the document
tab1 = Panel(child=column(button,p1), title="random")

p2 = figure(plot_width=300, plot_height=300)
p2.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=20, color="navy", alpha=0.5)
tab2 = Panel(child=p2, title="circle")

p3 = figure(plot_width=300, plot_height=300)
p3.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=3, color="navy", alpha=0.5)
tab3 = Panel(child=p3, title="line")

file_source = ColumnDataSource({'file_contents':[], 'file_name':[]})

def file_callback(attr,old,new):
    print( 'filename:', file_source.data['file_name'] )
    raw_contents = file_source.data['file_contents'][0]
    # remove the prefix that JS adds  
    prefix, b64_contents = raw_contents.split(",", 1)
    file_contents = base64.b64decode(b64_contents)
    file_io = StringIO(file_contents)
    print( "file contents:" )

file_source.on_change('data', file_callback)

trainsetselbut = Button(label='Upload',button_type='success')
trainsetselbut.callback = CustomJS(args=dict(file_source=file_source), code = """
function read_file(filename) {
    var reader = new FileReader();
    reader.onload = load_handler;
    reader.onerror = error_handler;
    // readAsDataURL represents the file's data as a base64 encoded string
    reader.readAsDataURL(filename);
}

function load_handler(event) {
    var b64string = event.target.result;
    file_source.data = {'file_contents' : [b64string], 'file_name':[input.files[0].name]};
    file_source.trigger("change");
}

function error_handler(evt) {
    if(evt.target.error.name == "NotReadableError") {
        alert("Can't read file!");
    }
}


var input = document.createElement('input');
input.setAttribute('type','file');
input.onchange = function(){
    if (window.FileReader) {
        read_file(input.files[0]);
    } else {
        alert('FileReader is not supported in this browser');
    }
}
input.click();
""")
tab4 = Panel(child=trainsetselbut, title='Training')

tabs = Tabs(tabs=[ tab1, tab2, tab3, tab4 ])

curdoc().add_root(tabs)
