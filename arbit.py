matplotlib.rcParams.update({'font.size': 8})    
plt.close(1)
plt.figure(1, figsize=(7.48, 7))
gs = gridspec.GridSpec(2, 4, width_ratios=[1,1,0.25, 0.1], wspace=0,  )
ax = []
axr = []
levs = np.arange(-2.2, 3.1, 0.2)

items = [acrossASF.break_28_CT, acrossASF.break_281_CT, acrossASF.break_2827_CT, acrossASF.break_gt_2827_CT]

for element in items:
    dfsel = ~element.isnull() & ~acrossASF.SHELF_BREAK_LONGITUDE.isnull() & ~acrossASF.month.isnull()
    lon_line = np.linspace(-180, 180.0000000001, 1000)
    month_line = np.arange(1,13,1)
    lon_grid, month_grid = np.meshgrid(lon_line, month_line)
    gnDepth_gridded = griddata(np.array([acrossASF[dfsel].SHELF_BREAK_LONGITUDE, acrossASF[dfsel].month]).T, 
                           element[dfsel], (lon_grid, month_grid))
    ax.append(plt.subplot(gs[0,0]))
    CF = ax[-1].contourf(lon_grid, month_grid, gnDepth_gridded, cmap="viridis", vmin=-2.2, vmax=3, extend='both', levels=levs)

subplot_titles = ["$\gamma^n=28$", "$\gamma^n=28.1$", "$\gamma^n=28.27$", "$\gamma^n>28.27$"]

ax_colorbar = plt.subplot(gs[:, 3])
cbr = Colorbar(ax = ax_colorbar, mappable = CF, orientation = 'vertical')
cbr.ax.set_ylabel("Depth")

subplot_titles = ["$\gamma^n=28$", "$\gamma^n=28.1$", "$\gamma^n=28.27$", "$\gamma^n>28.27$"]
for i in range(4):
    ax[i].set_xlim(-140, 190)
    ax[i].set_title(subplot_titles[i])
for i in range(1,4,2):
    ax[i].set_yticklabels("")
    ax[i].set_yticks([])
for i in range(2):
    ax[i].set_xticklabels("")
    ax[i].set_xticks([])

plt.show()
