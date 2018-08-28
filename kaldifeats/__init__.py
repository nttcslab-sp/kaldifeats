import pkg_resources
try:
    __version__ = pkg_resources.get_distribution('kaldifeats').version
except:
    __version__ = None
del pkg_resources
