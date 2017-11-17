
pro xdprob,infits,outfits,mode=mode
	if not keyword_set(mode) then mode=0
	simphot = mrdfits(infits,1)
	hdr = headfits(infits,exten=1)
	xdstruct = {PSFFLUX:fltarr(5), $
	            PSFFLUX_IVAR:fltarr(5), $
	            EXTINCTION:fltarr(5)}
	nQSO = size(simphot,/dim)
	xdinput = replicate(xdstruct,nQSO)
	sdssbands = [ 'SDSS-Legacy-u','SDSS-Legacy-g','SDSS-Legacy-r', $
	              'SDSS-Legacy-i','SDSS-Legacy-z' ]
	simbands = strsplit(sxpar(hdr,'OBSBANDS'),',',/extract)
	for i=0,4 do begin
		j = where(simbands eq sdssbands[i])
		xdinput.PSFFLUX[i,*] = simphot.obsFlux[j,*]
		xdinput.PSFFLUX_IVAR[i,*] = simphot.obsFluxErr[j,*]^(-2)
	endfor
	if mode eq 0 then begin
		print,'calculating eBOSS target selection probabilities...'
		xd = xdqsoz_calculate_prob(xdinput,0.9,6.0,/dereddened)
	endif else if mode eq 1 then begin
		print,'calculating BOSS target selection probabilities...'
		xd = xdqso_calculate_prob(xdinput,/dereddened)
	endif else begin
		print,'oops!'
		return
	endelse
	mwrfits,xd,outfits,/create
end
