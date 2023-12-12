import numpy as np 
import matplotlib.pyplot as plt


q0_1 = [0.47933532168726034 ,0.4810396250532595 ,0.6207925010651896 ,0.6489135066041756 ,0.6629740093736685 ,0.6685129953131658 ,0.6761823604601619 ,0.6787388155091606 ,
0.6923732424371538 ,0.6731998295696634 ,0.6787388155091606 ,0.691947166595654 ,0.6864081806561568 ,0.7230507030251385 ,0.7234767788666383 ,0.6812952705581593 ,0.6770345121431615 ,
0.6983383042181508 ,0.7034512143161483 ,0.6983383042181508 ,0.6868342564976566 ,0.7209203238176395 ,0.4870046868342565 ,0.4797613975287601 ,0.5027694929697486 ,0.5977844056242011 ,0.6169578184916915 ,
0.6442266723476778 ,0.6770345121431615 ,0.6821474222411589 ,0.674051981252663 ,0.7008947592671495 ,0.6957818491691521 ,0.7004686834256497 ,0.7072858968896464 ,0.7200681721346399 ,0.7213463996591394 ,
0.7281636131231359 ,0.7209203238176395 ,0.7281636131231359 ,0.7158074137196421 ,0.7324243715381338 ,0.7285896889646357 ,0.7405198125266298 ,0.7145291861951427 ,0.7392415850021303 ,0.7396676608436301 ,0.7405198125266298 ,0.7354069024286323 ,0.7243289305496379]
q0_2 = [0.4171282488282914 ,0.47933532168726034 ,0.561994034938219 ,.6463570515551769 ,0.5832978270132083 ,0.6548785683851726 ,0.6791648913506604 ,0.6945036216446527 ,0.677886663826161 ,0.6804431188751597 ,0.6736259054111632 ,0.6962079250106519 ,0.6838517256071581 ,
0.691947166595654 ,0.6864081806561568 ,0.7328504473796336 ,0.7153813378781423 ,0.7264593097571368 ,0.6876864081806562 ,0.6574350234341713 ,0.6774605879846612 ,0.6923732424371538 ,0.706007669365147 ,0.7072858968896464 ,0.6923732424371538 ,0.7030251384746485 ,
0.7128248828291436 ,0.7183638687686408 ,0.7187899446101406 ,0.7247550063911377 ,0.726033233915637 ,0.7264593097571368 ,0.7256071580741372 ,0.7136770345121431 ,0.7247550063911377 ,0.7277375372816361 ,0.6915210907541542 ,0.717937792927141 ,0.7213463996591394 ,
0.7337025990626331 ,0.7217724755006392 ,0.6761823604601619 ,0.7281636131231359 ,0.5973583297827013 ,0.697912228376651 ,0.7371112057946314 ,0.7025990626331488 ,0.7328504473796336 ,0.691947166595654 ,0.7068598210481466 ]
q0_3 = [0.4835960801022582 ,0.527907967618236 ,0.5636983383042181 ,0.6097145291861952 ,0.6442266723476778 ,0.6591393268001704 ,0.6693651469961653 ,0.6855560289731573 ,0.6953557733276523 ,0.6834256497656583 ,0.708138048572646 ,
0.6774605879846612 ,0.7030251384746485 ,0.7213463996591394 ,0.7337025990626331 ,0.722198551342139 ,0.7264593097571368 ,0.7281636131231359 ,0.7371112057946314 ,0.7290157648061355 ,0.7106945036216447 ,0.7366851299531316 ,0.7477631018321261 ,
0.7486152535151257 ,0.7413719642096294 ,0.7481891776736259 ,0.7524499360886238 ,0.7396676608436301 ,0.7541542394546229 ,0.755858542820622 ,0.7396676608436301 ,0.7562846186621218 ,0.7383894333191308 ,0.7562846186621218 ,0.7571367703451214 ,
0.7618236046016191 ,0.7584149978696207 ,0.6689390711546656 ,0.7417980400511291 ,0.7192160204516403 ,0.7294418406476353 ,0.7533020877716233 ,0.749893481039625 ,0.7383894333191308 ,0.7507456327226246 ,0.7533020877716233 ,0.752023860247124 ,0.7413719642096294 ,0.7567106945036216 ,0.7507456327226246 ]
q0_4= [0.47890924584576056 ,0.49893481039625054 ,0.5632722624627183 ,0.5994887089902002 ,0.5828717511717085 ,0.6327226246271836 ,0.6408180656156796 ,0.677886663826161 ,0.6808691947166595 ,0.6812952705581593 ,0.6936514699616532 ,0.6953557733276523 ,
0.6970600766936514 ,0.7008947592671495 ,0.7004686834256497 ,0.7047294418406477 ,0.7072858968896464 ,0.706007669365147 ,0.7008947592671495 ,0.7089902002556455 ,0.7153813378781423 ,0.7256071580741372 ,0.726033233915637 ,0.7268853855986366 ,0.7413719642096294 ,
0.7439284192586281 ,0.7435023434171283 ,0.7315722198551342 ,0.7443544951001279 ,0.7256071580741372 ,0.7388155091606305 ,0.7337025990626331 ,0.7494674051981253 ,0.7490413293566255 ,0.749893481039625 ,0.7426501917341287 ,0.749893481039625 ,0.746058798466127 ,
0.755858542820622 ,0.7575628461866212 ,0.7332765232211333 ,0.7541542394546229 ,0.7562846186621218 ,0.7596932253941202 ,0.7515977844056242 ,0.7490413293566255 ,0.7515977844056242 ,0.7682147422241159 ,0.7618236046016191 ,0.7481891776736259 ]
q0_5 = []


fedavg_1 = [0.4184064763527908 ,0.4797613975287601 ,0.49808265871325097 ,0.6297400937366852 ,0.6450788240306775 ,0.6659565402641671 ,0.6753302087771623 ,0.6812952705581593 ,0.6804431188751597 ,0.6736259054111632 ,0.6761823604601619 ,
0.6804431188751597 ,0.6838517256071581 ,0.6825734980826587 ,0.6898167873881551 ,0.6898167873881551 ,0.6949296974861525 ,0.6966340008521517 ,0.7030251384746485 ,0.7371112057946314 ,0.7392415850021303 ,0.7494674051981253 ,
0.7055815935236472 ,0.7562846186621218 ,0.7528760119301235 ,0.7545803152961227 ,0.763953983809118 ,0.6621218576906689 ,0.6536003408606732 ,0.722198551342139 ,0.7567106945036216 ,0.6224968044311887 ,0.657861099275671 ,0.691947166595654 ,
0.7298679164891351 ,0.7341286749041329 ,0.735832978270132 ,0.7264593097571368 ,0.7481891776736259 ,0.6957818491691521 ,0.752023860247124 ,0.7473370259906263 ,0.7405198125266298 ,0.6233489561141883 ,0.6638261610566681 ,0.6476352790796762 ,0.6825734980826587 ,0.7281636131231359 ,0.6314443971026843 ,0.7290157648061355]
fedavg_2 = [0.47251810822326373 ,0.5249254367277375 ,0.6097145291861952 ,0.6369833830421815 ,0.6595654026416702 ,0.6731998295696634 ,0.6714955262036643 ,0.6804431188751597 ,0.6868342564976566 ,0.6783127396676608 ,0.6817213463996591 ,0.6936514699616532 ,
0.6910950149126545 ,0.7072858968896464 ,0.7030251384746485 ,0.7072858968896464 ,0.7204942479761397 ,0.711972731146144 ,0.7243289305496379 ,0.731998295696634 ,0.6838517256071581 ,0.6851299531316575 ,0.6663826161056668 ,0.6761823604601619 ,0.7004686834256497 ,
0.7094162760971453 ,0.7204942479761397 ,0.7311461440136344 ,0.7383894333191308 ,0.7337025990626331 ,0.7239028547081381 ,0.7515977844056242 ,0.7439284192586281 ,0.7503195568811248 ,0.7486152535151257 ,0.7524499360886238 ,0.7494674051981253 ,0.7545803152961227 ,
0.7524499360886238 ,0.7596932253941202 ,0.7550063911376225 ,0.7605453770771198 ,0.7554324669791223 ,0.757988922028121 ,0.7656582871751172 ,0.7417980400511291 ,0.7515977844056242 ,0.6395398380911802 ,0.7605453770771198 ,0.7613975287601193]
fedavg_3 = [0.4801874733702599 ,0.5556028973157222 ,0.6126970600766937 ,0.6391137622496804 ,0.6523221133361738 ,0.6795909671921602 ,0.6736259054111632 ,0.6915210907541542 ,0.6851299531316575 ,0.7030251384746485 ,0.7077119727311462 ,0.7149552620366425 ,
0.7094162760971453 ,0.7264593097571368 ,0.726033233915637 ,0.7349808265871325 ,0.7354069024286323 ,0.7383894333191308 ,0.7413719642096294 ,0.7375372816361312 ,0.7400937366851299 ,0.7511717085641244 ,0.7337025990626331 ,0.7435023434171283 ,0.7503195568811248 ,
0.7315722198551342 ,0.7422241158926289 ,0.7405198125266298 ,0.7447805709416276 ,0.7490413293566255 ,0.7426501917341287 ,0.7435023434171283 ,0.7477631018321261 ,0.7469109501491266 ,0.7392415850021303 ,0.7481891776736259 ,0.7464848743076268 ,0.7362590541116318 ,
0.7405198125266298 ,0.7217724755006392 ,0.7354069024286323 ,0.7268853855986366 ,0.7354069024286323 ,0.7285896889646357 ,0.7345547507456327 ,0.7396676608436301 ,0.7234767788666383 ,0.7337025990626331 ,0.7362590541116318 ,0.7392415850021303 ]
fedavg_4 = [0.4801874733702599 ,0.4797613975287601 ,0.5325948018747337 ,0.5803152961227098 ,0.5466553046442266 ,0.6135492117596932 ,0.6148274392841926 ,0.6182360460161909 ,0.6220707285896889 ,0.6161056668086919 ,0.6237750319556881 ,0.61994034938219 ,
0.6276097145291862 ,0.6186621218576907 ,0.6237750319556881 ,0.6182360460161909 ,0.6063059224541969 ,0.5492117596932254 ,0.5837239028547081 ,0.6408180656156796 ,0.5262036642522369 ,0.5585854282062207 ,0.6787388155091606 ,0.6501917341286749 ,0.694077545803153 ,
0.7111205794631444 ,0.6020451640391989 ,0.6953557733276523 ,0.7008947592671495 ,0.7324243715381338 ,0.5713677034512143 ,0.7281636131231359 ,0.7447805709416276 ,0.7477631018321261 ,0.7456327226246272 ,0.7567106945036216 ,0.7490413293566255 ,0.7511717085641244 ,
0.7490413293566255 ,0.7477631018321261 ,0.757988922028121 ,0.7443544951001279 ,0.5641244141457179 ,0.7162334895611419 ,0.737963357477631 ,0.7239028547081381 ,0.7328504473796336 ,0.7226246271836387 ,0.7524499360886238 ,0.7669365146996165 ]
fedavg_5 = []


print(np.mean( np.array([ q0_1, q0_2, q0_3 ]), axis=0 ))

x = range(1, 51)
fig1 = plt.figure()
plt.plot(x, q0_1, 'r', label = "run 1", linestyle = "dotted")
plt.plot(x, q0_2, 'r', label = "run 2", linestyle = "dotted")
plt.plot(x, q0_3, 'r', label = "run 3", linestyle = "dotted")
plt.plot(x, q0_4, 'r', label = "run 4", linestyle = "dotted")
#plt.plot(x, q0_5, 'r', label = "run 5", linestyle = "dotted")

plt.plot(x, fedavg_1, 'g', label = "run 1", linestyle = "dotted")
plt.plot(x, fedavg_2, 'g', label = "run 3", linestyle = "dotted")
plt.plot(x, fedavg_3, 'g', label = "run 4", linestyle = "dotted")
plt.plot(x, fedavg_4, 'g', label = "run 5", linestyle = "dotted")
#plt.plot(x, fedavg_5, 'g', label = "run 6", linestyle = "dotted")
plt.legend()

plt.show()