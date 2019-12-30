plt.close(1)
plt.figure(1, figsize=(190/25.4 , 4))
gs = gridspec.GridSpec(1, 3, wspace=0, hspace=0)

ax = plt.subplot(gs[0,0])
dfsel = df.SHELF_BREAK_PROFILE & (df.ECHODEPTH > -1500) #& sel_months(df, [12,1,2,3,4,5])
OHC_SB = OHC.compute_OHC(df[dfsel], ax, h_b=-250, ymax=7e9, lon_bins= np.arange(0,361,2.5))
ax.set_ylabel("Ocean Heat Content Anomaly (Jm$^{-2}$)")

dfsel = df.SHELF_BREAK_PROFILE & (df.ECHODEPTH < -1500) #& sel_months(df, [12,1,2,3,4,5])
ax = plt.subplot(gs[0,1])
OHC_2 = OHC.compute_OHC(df[dfsel], ax, h_b=-250, ymax=7e9, hide_yticks=True, lon_bins= np.arange(0, 361, 2.5))


ax = plt.subplot(gs[0,2])
OHC_diff = OHC_2[:, 0] - OHC_SB[:, 0]
lons = np.arange(0, 360.01, 5)
lons = (lons[1:] + lons[:-1])*0.5

ax.fill_between(np.sort(lons),  OHC_diff[np.argsort(lons)], 0, facecolor="coral", edgecolor="r", alpha=0.5)
ax.set_ylim(0, 7e9)
ax.set_yticklabels([])
ax.set_xticks(np.arange(0, 361, 60))
ax.set_xticks(np.arange(30, 361, 60), minor=True)
ax.set_xticklabels(np.arange(0, 361, 60), rotation=90)
ax.set_xlim(0, 360)
ax.grid()

plt.savefig("./Images/slopeFront/OHC.jpg", dpi=600, bbox_inches='tight')
