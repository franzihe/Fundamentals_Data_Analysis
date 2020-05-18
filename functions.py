# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from imports import (xr, np, plt, sns, ccrs, glob, stats)

# +
### Create dask cluster to work parallel in large datasets

from dask.distributed import Client
client = Client(n_workers=2, 
                threads_per_worker=2, 
                memory_limit='4GB',
                processes=False)
client
chunks={'time' : 10,}
client 


# +
# plot style

def plot_style():
 #   plt.style.use('ggplot')
    sns.set_context('notebook')
    sns.set(font = 'Serif', font_scale = 1.2, )
    sns.set_style('ticks', 
                  {'font.family':'serif', #'font.serif':'Helvetica'
                   'grid.linestyle': '--',
                   'axes.grid': True,
                  }, 
                   )
plot_style()


# -

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    """This creates a plot in PlateCarree"""
    
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)


def add_map_features(ax):
    """Then I don't need to add it manually every time!"""
    ax.coastlines()
    gl = ax.gridlines()
   # ax.add_feature(cy.feature.BORDERS);
   # gl = ax.gridlines(draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False


def read_CMIP(cmip_path, var, ):
    """Finds CMIP6 model outputs and returns a xarray.
    
    Parameters:
    -----------
    cmip_path   : pre-defined directory
    var         : xarray variable to be importet
    """

    fn_list = [ff for ff in glob(cmip_path + var + '_Amon_*.nc') if (int(ff[-9:-5])>1985)]
    fn_list.sort()

    if len(fn_list) > 0:
        fn = xr.open_mfdataset(fn_list, 
                           chunks = chunks, 
                           parallel = True, 
                           use_cftime = True,
                          ) 
        fn['time'] = fn.indexes['time'].to_datetimeindex()  # make time from object to datetime64[ns]    
        return(fn)


def plt_mean_std_var_skew_kurt(var):
    """Plots the mean, standard deviation, variance, skewness and kurtosis for a given variable.
    
    Parameters:
    -----------
    var         : xarray variable to be plotted
    """
    f, axsm = sp_map(3,2, figsize = [18,8])
    axs = axsm.flatten()

    stat = ['Mean', 'Std', 'Variance', 'Skewness', 'Kurtosis']
    # Mean
    var.mean('time', keep_attrs = True).plot(ax = axs[0],
                                            transform=ccrs.PlateCarree(), 
                                             robust=True)

    # STD
    var.std('time', keep_attrs = True).plot(ax = axs[1],
                                             transform=ccrs.PlateCarree(), 
                                             robust=True)

    # Var
    var.var('time', keep_attrs = True).plot(ax = axs[2],
                                             transform=ccrs.PlateCarree(),
                                             robust=True)

    # skewness
    var.reduce(stats.skew,dim=('time'),keep_attrs = True).plot(ax = axs[3],
                                                                transform=ccrs.PlateCarree(), 
                                                                robust=True)

    # kurtosis
    var.reduce(stats.kurtosis,dim=('time'),keep_attrs = True).plot(ax = axs[4],
                                                                transform=ccrs.PlateCarree(), 
                                                                robust=True)


    axs[5].axis('off')
    for ax, i in zip(axs,range(len(stat))):
        ax.coastlines()
        ax.set_title(stat[i])


    plt.tight_layout()


def plt_stat_season(variable, stat, starty, endy):
    """Plots the seasonal statistics of the chosen statistic
    
    Parameters:
    -----------
    variable    : xarray variable to be plotted
    stat        : defines the statistics to plot,
                  must be: 'mean', 'st', 'var', 'skew', 'kur' 
    starty      : string of analysis begin
    endy        : string of analysis end
    """

    f, axsm = sp_map(2,2, figsize = [18,8])
    axs = axsm.flatten()

    for sea, ax in zip(variable.groupby('time.season').sum('time').season, axs.flatten()):
        if stat == 'mean':
            _title = 'Mean'
            im = variable.sel(time = (variable['time.season'] == sea), ).mean('time', keep_attrs = True).plot(ax = ax,
                                                                                                    transform=ccrs.PlateCarree(), 
                                                                                                    robust=True,
                                                                                                    add_colorbar=False)
        if stat == 'std':
            _title = 'Standard deviation'
            im = variable.sel(time = (variable['time.season'] == sea), ).std('time', keep_attrs = True).plot(ax = ax,
                                                                                                    transform=ccrs.PlateCarree(), 
                                                                                                    robust=True,
                                                                                                    add_colorbar=False)

        if stat == 'var':
            _title = 'Variance'
            im = variable.sel(time = (variable['time.season'] == sea), ).var('time', keep_attrs = True).plot(ax = ax,
                                                                                                    transform=ccrs.PlateCarree(), 
                                                                                                    robust=True,
                                                                                                    add_colorbar=False)
        if stat == 'skew':
            _title = 'Skewness'
            im = variable.sel(time = (variable['time.season'] == sea), ).reduce(stats.skew,dim=('time'),keep_attrs = True).plot(ax = ax,
                                                                                                                                transform=ccrs.PlateCarree(),
                                                                                                                                robust=True,
                                                                                                                                add_colorbar=False)

        if stat == 'kur':
            _title = 'Kurtosis'
            im = variable.sel(time = (variable['time.season'] == sea), ).reduce(stats.kurtosis,dim=('time'),keep_attrs = True).plot(ax = ax,
                                                                                                                                    transform=ccrs.PlateCarree(),
                                                                                                                                    robust=True,
                                                                                                                                    add_colorbar=False)

        ax.coastlines()

    f.subplots_adjust(right=0.8, top=0.9)
    f.suptitle('%s %s - %s' %(_title, starty, endy), fontweight='bold')
    cbar_ax = f.add_axes([0.85, 0.15, 0.025, 0.7])
    f.colorbar(im, 
               cax=cbar_ax, 
               extend = 'both',
               format='%.0e').ax.set_ylabel(variable.attrs['long_name'] + ' [' + variable.attrs['units'] + ']')
    plt.tight_layout()


def plt_twodhist_season(x, y, starty, endy, bins=None, cmap=None, range=None, norm = None):
    """ Plots a two-dimensional histogram of variable x and y.
    
    Parameters:
    -----------
    x           : variable on x-axis
    y           : variable on y-axis
    starty      : string of analysis begin
    endy        : string of analysis end
    bisn        : None or int or [int, int] or array-like or [array, array]
    cmap        : Colormap or str
    range       : array-like shape(2, 2), optional,
    norm        : Normalize, optional
    """
    
    f, axs = plt.subplots(2,2, figsize=[10,10], sharex=True, sharey=True)

    for sea, ax in zip(x.groupby('time.season').sum('time').season, axs.flatten()):

    
        _pp = np.asarray(x.sel(time = (x['time.season'] == sea), ))
        _cl  = np.asarray(y.sel(time = (y['time.season'] == sea), ))
        _p = _pp[~np.isnan(_pp)] 
        _c = _cl[~np.isnan(_pp)]

        _p = _p[~np.isnan(_c)]
        _c = _c[~np.isnan(_c)]

        counts, xedges, yedges, im  = ax.hist2d(_p,#p.flatten(), # use .flatten to pass a (N,) shape array as requested by hist2d
                                                _c,#l.flatten(), 
                                                #bins=(20,50),
                                                bins=bins,
                                                density = False,
                                           #     density = True,  # If False, the default, returns the number of samples in each bin. If True, returns the probability density function at the bin, bin_count / sample_count / bin_area.
                                                cmap = cmap,
                                                range = range,
                                                cmin = 0.5,
                                                norm = norm
                                               )
        ax.set_title(sea.values)
        ax.set_xlabel('Precipitation')
        ax.set_ylabel('Mass Fraction of Cloud Liquid + Ice Water')

    f.subplots_adjust(top = .85, right=0.8)
    f.suptitle('2D Histogram ' + starty + '-' + endy, fontweight='bold')
    #f.suptitle("Precipitation vs. Cloud Mass")
    cbar_ax = f.add_axes([1.01, 0.15, 0.025, 0.7])
    cbar = f.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Counts')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(1,0))



    plt.tight_layout()
