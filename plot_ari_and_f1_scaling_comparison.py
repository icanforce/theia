from matplotlib import pyplot, rc

x = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115]
theia_f = [0.8416156670746633, 0.8276968932706636, 0.8164115714400192, 0.8114064230343299, 0.8130752142177086, 0.8063163089069824, 0.8074959612277867, 0.8047066256021479, 0.8041247561377282, 0.8039163799947074, 0.8025324917910299, 0.803215800003671, 0.8008164484492456, 0.8012185374534376, 0.8020768007933845, 0.8015151304154075, 0.8019445491531592, 0.8003325181017553, 0.8007511001543671, 0.8020434511619623, 0.8008676700386544, 0.8006138590485183, 0.8007990454841275]
theia_ari = [0.9790017446840811,0.9830282044007854,0.9598484878843974,0.9385540407778556,0.9139235342958001,0.9090385867053791,0.9284662196549611,0.9456913693446137,0.9435734732350254,0.9188731750742889,0.9333785771607618,0.9508251538408863,0.9224388227392109,0.9165518741870976,0.933507677313897,0.9220108977237308,0.9251099139090393,0.9268769926698681,0.917825481351393,0.9154116565051857,0.9219395749804288,0.9217072156015232,0.9161457699898913]

#0.09628008752735223
snmnmf_f = [0.5358910891089107,0.5169061094180815,(0.516906109418015+0.5052572940603031)/2,0.5052572940603021,0.5081027175268013,0.5047463464170602,0.5028395262047702,0.5041014659858342,0.5020152972166682,0.5051262264358946,0.5039671817042913,0.49792664946553616,0.4999065866771842,0.49760910242113804,0.502115398643227,0.5020190767945414,0.49861540967584284,0.5,0.5,0.5,0.5,0.5,0.5]
snmnmf_ari = [0.0,0.0642384303627188,0.2378785013791004,0.0,0.08348212543227444,0.0021190252065368276,0.011525593682542919,0.04380222378948627,0.2837457331780263,-0.05800564885704432,9.126029542179085e-05,0.016690739456496472,-0.026017182693708403,0.059708453605807205,0.0009909098359619857,0.047287592471898836,0.016427625680658565,0,0,0,0,0,0]


pimim_f = [0.5249376558603491, 0.5245996248737554, 0.5154286863001306, 0.5071284512135753, 0.5075073577093829, 0.5075139923317614, 0.5110537356785539, 0.5043478260869565, 0.5070377985133637, 0.5075119696219249, 0.5015377801835493, 0.5037176427391222, 0.5020211023817913, 0.5000754773945202, 0.5017328809601983, 0.504491980468139, 0.4990066117317525, 0.4992992027489337, 0.4995820241561211, 0.5017956070265303,0.5,0.5,0.5]
pimim_ari = [0.13483209408213037, 0.1966810413264591, 0.08083987073966338, 0.06569393148766235, 0.08060698071271448, 0.03688355941264221, 0.04682561997769229, 0.02283001629058894, 0.027628461995355763, 0.025149270243239985, 0.034550657299996256, 0.0311016799286868, 0.02514103822594192, 0.021184525786108258, 0.028601051205904983, 0.01881611566010089, 0.01487361835257207, 0.02189750314345147, 0.016172479629666572, 0.014439835789225591, 0, 0, 0]

rc('font', **{'family': 'serif', 'serif': ['CMU Serif']})
rc('text', usetex=True)

f, ((ax1, ax3), (ax2, ax4)) = pyplot.subplots(2, 2, sharex=True, figsize=(4.5, 1.75))
f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.3, hspace=0.1)

for ax in (ax1, ax2):
    ax.plot(x, theia_ari, label='Theia')
    ax.plot(x, snmnmf_ari, label='SNMNMF')
    ax.plot(x, pimim_ari, label='PIMiM')

ax1.spines['bottom'].set_visible(False)
ax1.tick_params(axis='x', which='both', labeltop=False, bottom=False, labelbottom=False)
ax1.set_ylim([0.89, 1.0])
ax1.yaxis.set_ticks([0.9, 1.0])
ax1.set_title(r'E{f}{f}ect of $K$ on ARI')

ax2.spines['top'].set_visible(False)
ax2.xaxis.tick_bottom()
ax2.set_ylim([-0.1, 0.33])
ax2.yaxis.set_ticks([0.0, 0.2])

d = 0.015
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs)
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax2.transAxes)
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

for ax in (ax3, ax4):
    ax.plot(x, theia_f, label='Theia')
    ax.plot(x, snmnmf_f, label='SNMNMF')
    ax.plot(x, pimim_f, label='PIMiM')

ax3.spines['bottom'].set_visible(False)
ax3.tick_params(axis='x', which='both', labeltop=False, bottom=False, labelbottom=False)
ax3.set_ylim([0.79, 0.85])
ax3.yaxis.set_ticks([0.8, 0.85])
ax3.set_title(r'E{f}{f}ect of $K$ on $\mathrm{F_1}$ Score')
ax4.spines['top'].set_visible(False)
ax4.xaxis.tick_bottom()
ax4.set_ylim([0.49, 0.56])
ax4.yaxis.set_ticks([0.5, 0.55])

d = 0.015
kwargs = dict(transform=ax3.transAxes, color='k', clip_on=False)
ax3.plot((-d, +d), (-d, +d), **kwargs)
ax3.plot((1 - d, 1 + d), (-d, +d), **kwargs)
kwargs.update(transform=ax4.transAxes)
ax4.plot((-d, +d), (1 - d, 1 + d), **kwargs)
ax4.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

ax = f.add_subplot(1, 2, 1, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel('Number of modules', labelpad=15)
ax.set_ylabel('ARI', labelpad=15)

ax = f.add_subplot(1, 2, 2, frameon=False)
ax.tick_params(which='both', bottom=False, left=False)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_xlabel(r'Number of modules', labelpad=15)
ax.set_ylabel(r'$\mathrm{F_1}$ Score', labelpad=20)
ax1.legend(loc=(0.2, -2), ncol=3)

pyplot.savefig('ARI_and_F1_scaling_comparison.eps', bbox_inches='tight', dpi=100)
