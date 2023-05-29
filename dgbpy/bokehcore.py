import bokeh
from bokeh.core import enums
from bokeh.plotting import figure
from bokeh.io import curdoc
from bokeh.models import Spacer, ColumnDataSource, Range1d, Spinner

version = bokeh.__version__

def is_version_3():
    return version.startswith('3')

if is_version_3():
    from bokeh.models import TabPanel
    from bokeh.models import Select, Tabs, CheckboxGroup, Slider, RadioGroup, Button, MultiSelect, RangeSlider
else:
    from bokeh.models.widgets import Panel as TabPanel
    from bokeh.models.widgets import Select, Tabs, CheckboxGroup, Slider, RadioGroup, Button, MultiSelect, RangeSlider
    from bokeh.plotting import curdoc

def add_property(widget, name, value):
    if is_version_3():
        setattr(widget, name, value)
    else:
        setattr(widget, name, value)
    
def column(*args, **kwargs):
    from bokeh.layouts import column
    try:
        return column(*args, **kwargs)
    except ValueError:
        args = [arg() if callable(arg) else arg for arg in args]
        kwargs = {k: v() if callable(v) else v for k, v in kwargs.items()}
        return column(*args, **kwargs)
    
def row(*args, **kwargs):
    from bokeh.layouts import row
    try:
        return row(*args, **kwargs)
    except ValueError:
        args = [arg() if callable(arg) else arg for arg in args]
        kwargs = {k: v() if callable(v) else v for k, v in kwargs.items()}
        return row(*args, **kwargs)
    
    

class Div:
    def __init__(self, text='', **kwargs):
        self.isV3 = is_version_3() 
        if self.isV3:
            from bokeh.models import Div
            self.widget = Div( text=text, **kwargs)
        else:
            from bokeh.models.widgets import Div
            self.widget = Div( text=text, **kwargs)

    def __setattr__(self, name, value):
        try: 
            if name == 'styles' and not self.isV3:
                setattr(self.widget, 'style', value)
            else:
                setattr(self.widget, name, value)
        except:
            super().__setattr__(name, value)

    def __call__(self):
        return self.widget
