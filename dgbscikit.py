
platform = ('scikit','Scikit-learn')

def getMLPlatform():
  return platform[0]

def getUIMLPlatform():
  return platform[1]

scikit_dict = {
  'nb': 3
}

def getParams( nb=scikit_dict['nb'] ):
  return {
    'decimation': False,
    'number': nb
  }

