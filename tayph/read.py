
import pkg_resources
import os
import pdb
from astropy.io import fits
import astropy.constants as const
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import sys
import tayph.util as ut
from tayph.vartests import typetest,dimtest
import tayph.tellurics as mol
import tayph.system_parameters as sp
import tayph.functions as fun
import tayph.operations as ops
from tayph.operations import blaze_model, make_same_shape,average_overlapping_regions,spec_stich_n_norm
from tayph.ccf import xcor
import copy
import scipy.interpolate as interp
import pickle
from pathlib import Path
import warnings
import glob
from scipy import interpolate
import tayph.masking as masking
import subprocess
import textwrap
from astropy.utils.data import download_file
from .phoenix import get_phoenix_wavelengths, get_phoenix_model_spectrum
from scipy.ndimage import uniform_filter1d

#WHEN DESIGNING A NEW READ FUNCTION FOR AN INSTRUMENT, TAKE CARE THAT:
#1) EVERYTHING YOU READ FROM THE HEADERS IS CONVERTED TO FLOATS, INTS OR STRs WHERE NEEDED.
#(sometimes things end up as strings even though they should be floats, do it explicitly).
#2) THE WAVELENGTHS OF E2DS FRAMES NEED TO BE RETURNED IN NM. THE WAVELENGTHS OF S1Ds HOWEVER,
#SHOULD BE IN ANGSTROM..! (I don't like this, and this should be up for modification, but so be it).



__all__ = [
    "read_harpslike",
    "read_espresso",
    "read_uves",
    "read_spirou",
    "read_gianob",
    "read_crires",
    "read_fies",
    "read_hires_makee",
    "read_foces",
    "read_maroonx"
]



def read_harpslike(inpath,filelist,mode,read_s1d=True):
    """
    This reads a folder of HARPS or HARPSN data. Input is a list of filepaths and the mode (HARPS
    or HARPSN).
    """

    if mode=='HARPS':
        catkeyword = 'HIERARCH ESO DPR CATG'
        bervkeyword = 'HIERARCH ESO DRS BERV'
        thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
        Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
        Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    elif mode=='HARPSN':
        catkeyword = 'OBS-TYPE'
        bervkeyword = 'HIERARCH TNG DRS BERV'
        thfilekeyword = 'HIERARCH TNG DRS CAL TH FILE'
        Zstartkeyword = 'AIRMASS'
        Zendkeyword = 'AIRMASS'#These are the same because HARPSN doesnt have start and end keywords.
        #Down there, the airmass is averaged, so there is no problem in taking the average of the same number.
    else:
        raise ValueError(f"Error in read_harpslike: mode should be set to HARPS or HARPSN ({mode})")

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].endswith('e2ds_A.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[0].data)
            hdr = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr['NAXIS1'])
                norders=np.append(norders,hdr['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                # if nowave == True:
                # wavefile_used.append(hdr[thfilekeyword])
                #Record which wavefile was used by the pipeline to
                #create the wavelength solution.
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode=mode)/10.0#convert to nm.
                wave.append(wavedata)
                # if filelist[i].endswith('wave_A.fits'):
                #     print(filelist[i]+' (wave)')
                #     if nowave == True:
                #         warnings.warn(" in read_e2ds: nowave was set to True but a wave_A file was detected. This wave file is now ignored in favor of the header.",RuntimeWarning)
                #     else:
                #         wavedata=fits.getdata(inpath/filelist[i])
                #         wave.append(wavedata)

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('e2ds_A.fits','s1d_A.fits'))
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
        # if filelist[i].endswith('s1d_A.fits'):
                    hdul = fits.open(s1d_path)
                    data_1d = copy.deepcopy(hdul[0].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
            # if hdr[catkeyword] == 'SCIENCE':
                    s1d.append(data_1d)
                    if mode == 'HARPSN':#In the case of HARPS-N we need to convert the units of the
                        #elevation and provide a UTC keyword.
                        hdr1d['TELALT'] = np.degrees(float(hdr1d['EL']))
                        hdr1d['UTC'] = (float(hdr1d['MJD-OBS'])%1.0)*86400.0
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_harpslike(): BERV correction of s1d file is not'
                        f'equal to that of the e2ds file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)#Doppler factor BERV.
                    wave1d.append((hdr1d['CDELT1']*np.arange(len(data_1d), dtype=float)+hdr1d['CRVAL1'])*gamma)

    #Check that all exposures have the same number of pixels, and clip s1ds if needed.
    # min_npx1d = int(np.min(np.array(npx1d)))
    # if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
    #     warnings.warn("in read_e2ds when reading HARPS data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest length.",RuntimeWarning)
    #     for i in range(len(s1d)):
    #         wave1d[i]=wave1d[i][0:min_npx1d]
    #         s1d[i]=s1d[i][0:min_npx1d]
    #         npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
    'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
    'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)

def read_carmenes(inpath,filelist,channel,construct_s1d=True):
    """
    This reads a folder of CARMENES visible (VIS) or infra-red (NIR) channel data. Input is a list
    of filepaths and the mode ('VIS' or 'NIR').
    """



    catkeyword = 'HIERARCH CAHA INS ICS IMAGETYP'
    bervkeyword = 'HIERARCH CARACAL BERV'
    thfilekeyword = 'HIERARCH CARACAL WAVE FILE'
    Zstartkeyword = 'AIRMASS'
    Zendkeyword = 'AIRMASS'#These are the same because CARMENES doesnt have start and end keywords.

    if channel not in ['vis','nir']:
        raise ValueError(f"Error in read_carmenes: channel should be set to VIS or NIR ({channel})")

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    blaze=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].endswith('-'+channel.lower()+'_A.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[1].data)
            cont = copy.deepcopy(hdul[2].data)
            sigma = copy.deepcopy(hdul[3].data)
            wavedata = copy.deepcopy(hdul[4].data)/10.0
            hdr = hdul[0].header
            spechdr = hdul[1].header
            hdul.close()
            del hdul[1].data
            del hdul[2].data
            del hdul[3].data
            del hdul[4].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,spechdr['NAXIS1'])
                norders=np.append(norders,spechdr['NAXIS2'])
                e2ds.append(data)
                blaze.append(data/sigma**2)
                berv=np.append(berv,hdr[bervkeyword])
                airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
                wave.append(ops.vactoair(wavedata))


                if construct_s1d:
                    wave_1d, data_1d = spec_stich_n_norm(data,wavedata,cont,sigma)

                    s1d.append(data_1d)

                    hdr1d = copy.deepcopy(hdr)
                    hdr1d['UTC'] = (float(hdr1d['MJD-OBS'])%1.0)*86400.0
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    berv1d = hdr1d[bervkeyword]
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)#Doppler factor BERV.
                    gamma = 1 #turning berv off
                    wave1d.append(ops.vactoair(wave_1d)*gamma*10)

    BLAZE_Model = blaze_model(np.nanmean(blaze, axis = 0)) #Calucalting blaze model
    e2ds = list(e2ds*BLAZE_Model[np.newaxis,:]) #Deblazing the data

    if construct_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass}
    return(output)



def read_uves(inpath,filelist,mode):
    """This reads a folder of UVES-blue or UVES-red intermediate pipeline products. Input is a list of filepaths and the mode."""
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    npx1d=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    catkeyword = 'HIERARCH ESO DPR CATG'
    bervkeyword = 'HIERARCH ESO DRS BERV'
    thfilekeyword = 'HIERARCH ESO DRS CAL TH FILE'
    Zstartkeyword = 'HIERARCH ESO TEL AIRM START'
    Zendkeyword = 'HIERARCH ESO TEL AIRM END'
    for i in range(len(filelist)):
        print(f'------{filelist[i]}', end="\r")
        if (inpath/Path(filelist[i])).is_dir():
            tmp_products = [i for i in (inpath/Path(filelist[i])).glob('resampled_science_*.fits')]
            tmp_products1d = [i for i in (inpath/Path(filelist[i])).glob('red_science_*.fits')]
            if mode == 'UVES-red' and len(tmp_products) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 resampled_science files (redl and redu), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 resampled_science files (blue), but {len(tmp_products)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-red' and len(tmp_products1d) != 2:
                raise ValueError(f"in read_e2ds: When mode=UVES-red there should be 2 red_science files (redl and redu), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")
            if mode == 'UVES-blue' and len(tmp_products1d) != 1:
                raise ValueError(f"in read_e2ds: When mode=UVES-rblue there should be 1 red_science files (blue), but {len(tmp_products1d)} were detected in {str(inpath/Path(filelist[i]))}.")

            data_combined = []#This will store the two chips (redu and redl) in case of UVES_red, or simply the blue chip if otherwise.
            wave_combined = []
            wave1d_combined=[]
            data1d_combined=[]
            norders_tmp = 0
            for tmp_product in tmp_products:
                hdul = fits.open(tmp_product)
                data = copy.deepcopy(hdul[0].data)
                hdr = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                wavedata=ut.read_wave_from_e2ds_header(hdr,mode='UVES')/10.0#Convert to nm.
                data_combined.append(data)
                wave_combined.append(wavedata)
                norders_tmp+=np.shape(data)[0]

            for tmp_product in tmp_products1d:
                hdul = fits.open(tmp_product)
                data_1d = copy.deepcopy(hdul[0].data)
                hdr1d = hdul[0].header
                hdul.close()
                del hdul[0].data
                if not hdr1d['HIERARCH ESO PRO SCIENCE']:#Only add if it's actually a science product:#I force the user to supply only science exposures in the input  folder. No BS allowed... UVES is hard enough as it is.
                    raise ValueError(f' in read_e2ds: UVES file {tmp_product} is not classified as a SCIENCE file, but should be. Remove it from the folder?')
                npx_1d = hdr1d['NAXIS1']
                wavedata = np.arange(npx_1d, dtype=float)*hdr1d['CDELT1']+hdr1d['CRVAL1']
                data1d_combined.append(data_1d)
                wave1d_combined.append(wavedata)

            if len(data_combined) < 1 or len(data_combined) > 2:#Double-checking that length here...
                raise ValueError(f'in read_e2ds(): Expected 1 or 2 chips, but {len(data_combined)} files were somehow read.')
            #The chips generally don't give the same size. Therefore I will pad the smaller one with NaNs to make it fit:
            if len(data_combined) != len(data1d_combined):
                raise ValueError(f'in read_e2ds(): The number of chips in the 1d and 2d spectra is not the same {len(data1d_combined)} vs {len(data_combined)}.')

            if len(data_combined) == 2:
                chip1 = data_combined[0]
                chip2 = data_combined[1]
                wave1 = wave_combined[0]
                wave2 = wave_combined[1]
                npx_1 = np.shape(chip1)[1]
                npx_2 = np.shape(chip2)[1]
                no_1 = np.shape(chip1)[0]
                no_2 = np.shape(chip2)[0]
                npx_max = np.max([npx_1,npx_2])
                npx_min = np.min([npx_1,npx_2])
                diff = npx_max-npx_min
                #Pad the smaller one with NaNs to match the wider one:
                if npx_1 < npx_2:
                    chip1=np.hstack([chip1,np.zeros((no_1,diff))*np.nan])
                    wave1=np.hstack([wave1,np.zeros((no_1,diff))*np.nan])
                else:
                    chip2=np.hstack([chip2,np.zeros((no_2,diff))*np.nan])
                    wave2=np.hstack([wave2,np.zeros((no_2,diff))*np.nan])
                #So now they can be stacked:
                e2ds_stacked = np.vstack((chip1,chip2))
                wave_stacked = np.vstack((wave1,wave2))
                if np.shape(e2ds_stacked)[1] != np.shape(wave_stacked)[1]:
                    raise ValueError("Width of stacked e2ds and stacked wave frame are not the same. Is the wavelength solution in the header of this file correct?")
                npx=np.append(npx,np.shape(e2ds_stacked)[1])

                e2ds.append(e2ds_stacked)
                wave.append(wave_stacked)
                chip1_1d = data1d_combined[0]
                chip2_1d = data1d_combined[1]
                wave1_1d = wave1d_combined[0]
                wave2_1d = wave1d_combined[1]
                if np.nanmean(wave1_1d) < np.nanmean(wave2_1d):
                    combined_data_1d = np.concatenate((chip1_1d,chip2_1d))
                    combined_wave_1d = np.concatenate((wave1_1d,wave2_1d))
                else:
                    combined_data_1d = np.concatenate((chip2_1d,chip1_1d))
                    combined_wave_1d = np.concatenate((wave2_1d,wave1_1d))
                wave1d.append(combined_wave_1d)
                s1d.append(combined_data_1d)
                npx1d=np.append(npx1d,len(combined_wave_1d))
            else:
                e2ds.append(data_combined[0])
                wave.append(wave_combined[0])
                npx=np.append(npx,np.shape(data_combined[0])[1])
                wave1d.append(wave1d_combined[0])
                s1d.append(data1d_combined[0])
                npx1d=np.append(npx1d,len(combined_wave_1d))
            #Only using the keyword from the second header in case of redl,redu.
            s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
            framename.append(hdr['ARCFILE'])
            header.append(hdr)
            obstype.append('SCIENCE')
            texp=np.append(texp,hdr['EXPTIME'])
            date.append(hdr['DATE-OBS'])
            mjd=np.append(mjd,hdr['MJD-OBS'])
            norders=np.append(norders,norders_tmp)
            airmass=np.append(airmass,0.5*(hdr[Zstartkeyword]+hdr[Zendkeyword]))#This is an approximation where we take the mean airmass.
            berv_i=sp.calculateberv(hdr['MJD-OBS'],[hdr['HIERARCH ESO TEL GEOLAT'],hdr['HIERARCH ESO TEL GEOLON'],hdr['HIERARCH ESO TEL GEOELEV']],hdr['RA'],hdr['DEC'],mode=mode)
            berv = np.append(berv,berv_i)
            hdr1d['HIERARCH ESO QC BERV']=berv_i#Append the berv here using the ESPRESSO berv keyword, so that it can be used in molecfit later.
            s1dhdr.append(hdr1d)

    #Check that all exposures have the same number of pixels, and clip orders if needed.
    min_npx = int(np.min(np.array(npx)))
    min_npx1d = int(np.min(np.array(npx1d)))
    if np.sum(np.abs(np.array(npx)-npx[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all e2ds files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The e2ds files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(e2ds)):
            wave[i]=wave[i][:,0:min_npx]
            e2ds[i]=e2ds[i][:,0:min_npx]
            npx[i]=min_npx
    if np.sum(np.abs(np.array(npx1d)-npx1d[0])) != 0:
        warnings.warn("in read_e2ds when reading UVES data: Not all s1d files have the same number of pixels. This could have happened if the pipeline has extracted one or two extra pixels in some exposures but not others. The s1d files will be clipped to the smallest width.",RuntimeWarning)
        for i in range(len(s1d)):
            wave1d[i]=wave1d[i][0:min_npx1d]
            s1d[i]=s1d[i][0:min_npx1d]
            npx1d[i]=min_npx1d
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,'npx1d':npx1d,'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)



def read_espresso(inpath,filelist,read_s1d=True,skysub=False):
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    SNR = []
    catkeyword = 'EXTNAME'
    bervkeyword = 'HIERARCH ESO QC BERV'
    airmass_keyword1 = 'HIERARCH ESO TEL'
    airmass_keyword2 = ' AIRM '
    airmass_keyword3_start = 'START'
    airmass_keyword3_end = 'END'
    snr_keyword = 'HIERARCH ESO QC ORDER102 SNR' #SNR in order 102, at 550nm.

    if skysub:
        type_suffix = 'S2D_SKYSUB_A.fits'
    else:
        type_suffix = 'S2D_BLAZE_A.fits'

    for i in range(len(filelist)):
        if filelist[i].endswith(type_suffix):
            hdul = fits.open(inpath/filelist[i])
            data = copy.deepcopy(hdul[1].data)
            hdr = hdul[0].header
            hdr2 = hdul[1].header
            wavedata=copy.deepcopy(hdul[5].data)
            hdul.close()
            del hdul

            if hdr2[catkeyword] == 'SCIDATA':
                # print('science keyword found')
                print(f'------{filelist[i]}', end="\r")
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append('SCIENCE')
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr2['NAXIS1'])
                norders=np.append(norders,hdr2['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])#in km.s.
                telescope = hdr['TELESCOP'][-1]
                airmass = np.append(airmass,0.5*(hdr[airmass_keyword1+telescope+' AIRM START']+hdr[airmass_keyword1+telescope+' AIRM END']))
                wave.append(wavedata/10.0)#*(1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value))
                #Ok.! So unlike HARPS, ESPRESSO wavelengths are actually BERV corrected in the S2Ds.
                #WHY!!!?. WELL SO BE IT. IN ORDER TO HAVE E2DSes THAT ARE ON THE SAME GRID, AS REQUIRED, WE UNDO THE BERV CORRECTION HERE.
                #WHEN COMPARING WAVE[0] WITH WAVE[1], YOU SHOULD SEE THAT THE DIFFERENCE IS NILL.
                #THATS WHY LATER WE CAN JUST USE WAVE[0] AS THE REPRESENTATIVE GRID FOR ALL.
                #BUT THAT IS SILLY. JUST SAVE THE WAVELENGTHS!
                SNR.append(hdr[snr_keyword])
                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('_'+type_suffix,'_S1D_A.fits'))
                    #Need the blazed files. Not the S2D_A's by themselves.
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    hdul = fits.open(s1d_path)
                    data_table = copy.deepcopy(hdul[1].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
                    s1d.append(data_table.field(2))

                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_espresso(): BERV correction of S1D file is not'
                        f'equal to that of the S2D file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)
                    wave1d.append(data_table.field(1)*gamma)#This is in angstroms.
                    #We need to check to which UT ESPRESSO was connected, so that we can read
                    #the weather information (which is UT-specific) and parse them into the
                    #header using UT-agnostic keywords that are in the ESPRESSO.par file.
                    TELESCOP = hdr1d['TELESCOP'].split('U')[1]#This is the number of the UT, either 1, 2, 3 or 4.
                    if TELESCOP not in ['1','2','3','4']:
                        raise ValueError(f"in read_e2ds when reading ESPRESSO data. The UT telescope is not recognised. (TELESCOP={hdr['TELESCOP']})")
                    else:
                        hdr1d['TELALT']     = hdr1d[f'ESO TEL{TELESCOP} ALT']
                        hdr1d['RHUM']       = hdr1d[f'ESO TEL{TELESCOP} AMBI RHUM']
                        hdr1d['PRESSURE']   = (hdr1d[f'ESO TEL{TELESCOP} AMBI PRES START']+
                                            hdr1d[f'ESO TEL{TELESCOP} AMBI PRES END'])/2.0
                        hdr1d['AMBITEMP']   = hdr1d[f'ESO TEL{TELESCOP} AMBI TEMP']
                        hdr1d['M1TEMP']     = hdr1d[f'ESO TEL{TELESCOP} TH M1 TEMP']
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
    if read_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd,'snr550':SNR}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'snr550':SNR}
    return(output)

def read_nirps(inpath,filelist,read_s1d=True,skysub=True):
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    catkeyword = 'EXTNAME'
    bervkeyword = 'HIERARCH ESO QC BERV'
    airmass_start = 'HIERARCH ESO TEL AIRM START'
    airmass_end = 'HIERARCH ESO TEL AIRM END'


    type_suffix = 'S2D_BLAZE_A.fits'

    for i in range(len(filelist)):
        if filelist[i].endswith(type_suffix):
            hdul = fits.open(inpath/filelist[i])
            wavedata=copy.deepcopy(hdul[5].data)
            if skysub:
                hdul2 = fits.open(str(inpath/filelist[i]).replace('BLAZE_A.fits','BLAZE_B.fits'))
                if np.sum(wavedata-hdul2[5].data) != 0:
                    raise Exception("Error in read_nirs: The wavelengths of two blaze files "
                    f"are not identical ({str(inpath/filelist[i])}). Sky subtraction cannot "
                    "proceed.")
                data = copy.deepcopy(hdul[1].data-hdul2[1].data)#This way of doing sky subtraction
                #is valid because the two blaze files have the same wavelength axis.
                del hdul2
            else:
                data = copy.deepcopy(hdul[1].data)
            hdr = hdul[0].header
            hdr2 = hdul[1].header

            hdul.close()
            del hdul

            if hdr2[catkeyword] == 'SCIDATA':
                # print('science keyword found')
                print(f'------{filelist[i]}', end="\r")
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append('SCIENCE')
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                npx=np.append(npx,hdr2['NAXIS1'])
                norders=np.append(norders,hdr2['NAXIS2'])
                e2ds.append(data)
                berv=np.append(berv,hdr[bervkeyword])#in km.s.
                telescope = hdr['TELESCOP'][-1]
                airmass = np.append(airmass,0.5*(hdr[airmass_start]+hdr[airmass_end]))
                wave.append(wavedata/10.0)#*(1.0-(hdr[bervkeyword]*u.km/u.s/const.c).decompose().value))
                #Ok.! So unlike HARPS, ESPRESSO wavelengths are actually BERV corrected in the S2Ds.
                #WHY!!!?. WELL SO BE IT. IN ORDER TO HAVE E2DSes THAT ARE ON THE SAME GRID, AS REQUIRED, WE UNDO THE BERV CORRECTION HERE.
                #WHEN COMPARING WAVE[0] WITH WAVE[1], YOU SHOULD SEE THAT THE DIFFERENCE IS NILL.
                #THATS WHY LATER WE CAN JUST USE WAVE[0] AS THE REPRESENTATIVE GRID FOR ALL.
                #BUT THAT IS SILLY. JUST SAVE THE WAVELENGTHS!

                if read_s1d:
                    if skysub:
                        s1d_path=inpath/Path(str(filelist[i]).replace('_'+type_suffix,
                        '_S1D_SKYSUB_A.fits'))
                        ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    else:
                        s1d_path=inpath/Path(str(filelist[i]).replace('_'+type_suffix,
                        '_S1D_A.fits'))
                        try:
                            ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                        except:
                            print('---Warning: S1D file not found. Trying Skysub instead.')
                            s1d_path=inpath/Path(str(filelist[i]).replace('_'+type_suffix,
                        '_S1D_SKYSUB_A.fits'))
                    #Need the blazed files. Not the S2D_A's by themselves.
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    hdul = fits.open(s1d_path)
                    data_table = copy.deepcopy(hdul[1].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
                    s1d.append(data_table.field(2))

                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_espresso(): BERV correction of S1D file is not'
                        f'equal to that of the S2D file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)
                    wave1d.append(data_table.field(1)*gamma)#This is in angstroms.
                    #We need to check to which UT ESPRESSO was connected, so that we can read
                    #the weather information (which is UT-specific) and parse them into the
                    #header using UT-agnostic keywords that are in the ESPRESSO.par file.

                    hdr1d['TELALT']     = hdr1d[f'ESO TEL ALT']
                    hdr1d['RHUM']       = hdr1d[f'ESO TEL AMBI RHUM']
                    hdr1d['PRESSURE']   = (hdr1d[f'ESO TEL AMBI PRES START']+
                                            hdr1d[f'ESO TEL AMBI PRES END'])/2.0
                    hdr1d['AMBITEMP']   = hdr1d[f'ESO TEL AMBI TEMP']
                    hdr1d['M1TEMP']     = hdr1d[f'ESO TEL AMBI TEMP']#THIS IS SET EQUAL TO AMBIENT
                    #BECAUSE M1 TEMP IS NOT IN THE FITS HEADER.
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
    if read_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass}
    return(output)







def read_spirou(inpath,filelist,read_s1d=False):
    """
    This reads a folder of SPIROU spectra expecting *t.fits for telluric reduce spectra to exists

    As this data is already telluric reduced no effort is done for creating the s1d data structures
    """

    if read_s1d:
        wrn_msg = ('read s1d files not implemented yet for SPIROU, read_s1d reset to false!')
        ut.tprint(wrn_msg)
        read_s1d = False

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    for i in range(len(filelist)):
        if filelist[i].endswith('t.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            fluxdata = copy.deepcopy(hdul[1].data)
            wavedata = ops.vactoair(copy.deepcopy(hdul[2].data))
            blazedata = copy.deepcopy(hdul[3].data)
            teldata = copy.deepcopy(hdul[4].data)
            hdr = hdul[0].header
            fluxhdr = hdul[1].header
            wavehdr = hdul[2].header
            blazehdr = hdul[3].header
            telhdr = hdul[4].header
            hdul.close()
            del hdul[1].data
            del hdul[2].data
            del hdul[3].data
            del hdul[4].data

            __npx = fluxhdr['NAXIS1']
            __norders = fluxhdr['NAXIS2']
            idxfilter = []
            for idx in range(__norders):
                numberofNaN = np.count_nonzero(np.isnan(fluxdata[idx]))
                idxfilter.append(numberofNaN != fluxdata[idx].size)
#                __ratio = float(numberofNaN) / float(fluxdata[idx].size)
#                idxfilter.append(__ratio < 1.0)
#                print("numberofNaN"+str(numberofNaN)+"  fluxdata:"+str(fluxdata[idx].size)+"   ratio:"+str(__ratio))
            fluxdata = fluxdata[idxfilter]
            wavedata = wavedata[idxfilter]
            blazedata = blazedata[idxfilter]
            teldata = teldata[idxfilter]
#            print("dropped count of orders: "+str(np.size(idxfilter) - np.sum(idxfilter)))
            __norders = __norders - np.size(idxfilter) + np.sum(idxfilter)  # subtracts number of False

            framename.append(filelist[i])
            header.append(hdr)
            obstype.append(hdr['OBSTYPE'])
            texp=np.append(texp,hdr['EXPTIME'])
            date.append(hdr['DATE-OBS']+'T'+hdr['UTC-OBS'])
            mjd=np.append(mjd,hdr['MJD-OBS'])
            npx=np.append(npx,__npx)
            norders=np.append(norders,__norders)
            e2ds.append(fluxdata)
            wave.append(wavedata)
            berv=np.append(berv,fluxhdr['BERV'])
            airmass=np.append(airmass,hdr['AIRMASS'])


    if read_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
              'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
              'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
              'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
              'norders':norders,'berv':berv,'airmass':airmass}
    return(output)



def read_gianob(inpath,filelist,read_s1d=True):
    """
    This reads a folder of SPIROU spectra expecting *t.fits for telluric reduce spectra to exists

    As this data is already telluric reduced no effort is done for creating the s1d data structures
    """
    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]

    for i in range(len(filelist)):
#        if filelist[i].endswith('AB_ms1d.fits'):
        if filelist[i].endswith('_A_ms1d.fits') or filelist[i].endswith('_B_ms1d.fits'):
            hdul = fits.open(inpath/filelist[i])
            fitsdata = copy.deepcopy(hdul[1].data)
            hdr = hdul[0].header
            hdul.close()
            del hdul[1].data

            if hdr['OBS-TYPE'] == 'SCIENCE':
                # print('science keyword found')
                print(f'------{filelist[i]}', end="\r")
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append('SCIENCE')
                texp=np.append(texp,hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                mjd=np.append(mjd,hdr['MJD-OBS'])
                berv=np.append(berv,hdr['HIERARCH TNG DRS BERV'])#in km.s.
                airmass = np.append(airmass,hdr['AIRMASS'])
                ordernumbers = []
                wavedata = []
                fluxdata = []
                snrdata = []
                __npx = len(fitsdata[0][1])
                __norders = len(fitsdata)
                npx=np.append(npx,__npx)
                norders=np.append(norders,__norders)
                for idx in range(len(fitsdata)):
                    ordernumbers.append(fitsdata[idx][0])
                    wavedata.append(fitsdata[idx][1])
                    fluxdata.append(fitsdata[idx][2])
                    snrdata.append(fitsdata[idx][3])
                wave.append(ops.vactoair(np.array(wavedata)))
                e2ds.append(np.array(fluxdata))
#                print("---wave then flux---")
#                print(np.array(wavedata))
#                print(np.array(fluxdata))

                if read_s1d:
                    s1d_path=inpath/Path(str(filelist[i]).replace('_ms1d.fits','_s1d.fits'))
                    #Need the blazed files. Not the S2D_A's by themselves.
                    ut.check_path(s1d_path,exists=True)#Crash if the S1D doesn't exist.
                    hdul1d = fits.open(s1d_path)
                    hdr1d = hdul1d[0].header
                    fluxdata1d = copy.deepcopy(hdul1d[0].data)
                    hdul1d.close()
                    del hdul1d[0].data

                    s1d.append(fluxdata1d)

                    bervkeyword = 'HIERARCH TNG DRS BERV'
                    berv1d = hdr1d[bervkeyword]
                    if berv1d != hdr[bervkeyword]:
                        wrn_msg = ('WARNING in read_gianob(): BERV correction of S1D file is not'
                        f'equal to that of the S2D file. {berv1d} vs {hdr[bervkeyword]}')
                        ut.tprint(wrn_msg)
                    gamma = (1.0-(berv1d*u.km/u.s/const.c).decompose().value)
                    crval = hdr1d['CRVAL1']
                    cdelt = hdr1d['CDELT1']
                    mywave = crval
                    wavedata1d = [mywave*gamma*10]
                    for idx in range(len(fluxdata1d)-1):
                        mywave += cdelt
                        wavedata1d.append(mywave*gamma*10)
                    wave1d.append(ops.vactoair(np.array(wavedata1d)))
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,hdr1d['MJD-OBS'])
                    hdr1d['TELALT'] = np.degrees(float(hdr1d['EL']))
                    hdr1d['UTC'] = (float(hdr1d['MJD-OBS']) % 1.0) * 86400.0
#                    print("wave:"+str(len(wavedata1d))+" flux:"+str(len(fluxdata1d)))

    if read_s1d:
        output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    else:
        output = {'wave':wave,'e2ds':e2ds,'header':header,
        'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
        'norders':norders,'berv':berv,'airmass':airmass}
    return(output)



def read_pycrires(inpath,filelist,rawpath=None,read_s1d=True,nod='both'):
    import os
    import glob
    import copy
    from pathlib import Path
    import pathlib
    import numpy as np
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation

    # The following variables define lists in which all the necessary data will be stored.

    if not rawpath:
        rawpath = ut.check_path(inpath/'../../raw/',exists=True) #This is the default when reading
        #pycrires data from the ..../products/obs_nodding folder. Set rawpath explicitly if reading
        #from a different folder structure.

    framename = []
    header = []
    obstype = []
    texp = np.array([])
    date = []
    mjd = np.array([])
    s1dmjd = np.array([])
    airmass = np.array([])
    berv = np.array([])

    npx = np.array([])
    norders = np.array([])
    out_wave = []
    out_e2ds = []
    out_wave1d = []
    out_s1d = []

    rawfiles_infileidx = {}
    rawfiles_mjdobs = {}
    for i in range(len(filelist)):
        nod_trigger = False#We skip this file unless this becomes true:
        if nod =='A' or nod =='B':#If we only want one nod:
            if filelist[i].startswith(f'cr2res_obs_nodding_extracted{nod}'):
                nod_trigger = True
        else:#If we want both nods at the same time:
            if filelist[i].startswith('cr2res_obs_nodding_extractedA') or filelist[i].startswith('cr2res_obs_nodding_extractedB'):
                nod_trigger = True
        reducedfile = inpath / filelist[i]
        if reducedfile.is_file() and nod_trigger:
            hdu = fits.open(reducedfile)
            rawfile = hdu[0].header['HIERARCH ESO PRO REC1 RAW1 NAME']
            # if filelist[i].startswith('cr2res_obs_nodding_extractedB'):
            #     rawfile = hdu[0].header['HIERARCH ESO PRO REC1 RAW2 NAME']
            rawfiles_infileidx[rawfile] = i
            hdu.close()
            hdu = fits.open(rawpath / rawfile)
            mjdobs = hdu[0].header['MJD-OBS']
            rawfiles_mjdobs[rawfile] = mjdobs
            hdu.close()
    rawfiles_sorted = dict(sorted(rawfiles_mjdobs.items(), key=lambda item: item[1])).keys()

    expno = -1
    for rawfile in rawfiles_sorted:
        expno=expno+1
        print(f"exposure no: {expno}")
        print(filelist[rawfiles_infileidx[rawfile]])
        print(rawfile)

        # Read in headers from the RAW file as the headers in the extracted files are not for the individual nodding frames
        hdu = fits.open(rawpath / rawfile)
        hdrraw = hdu[0].header
        hdu.close()

        airmass_calc = 0.5 * (float(hdrraw['HIERARCH ESO TEL AIRM START']) + float(hdrraw['HIERARCH ESO TEL AIRM END']))
        observatory = EarthLocation.from_geodetic(lat=hdrraw['HIERARCH ESO TEL GEOLAT'] * u.deg,
                                                  lon=hdrraw['HIERARCH ESO TEL GEOLON'] * u.deg,
                                                  height=hdrraw['HIERARCH ESO TEL GEOELEV'] * u.m)
        sc = SkyCoord(ra=hdrraw['RA'] * u.deg,
                      dec=hdrraw['DEC'] * u.deg)
        barycorr = sc.radial_velocity_correction(obstime=Time(hdrraw['MJD-OBS'], format='mjd'),
                                                 location=observatory).to(u.km / u.s)
        hdrraw['HIERARCH ESO QC BERV'] = barycorr.value
        hdrraw['HIERARCH ESO QC AIRMASS'] = airmass_calc

        framename.append(filelist[rawfiles_infileidx[rawfile]])
        header.append(hdrraw)
        obstype.append('SCIENCE')
        texp = np.append(texp, hdrraw['EXPTIME'])
        date.append(hdrraw['DATE-OBS'])
        mjd = np.append(mjd, hdrraw['MJD-OBS'])
        s1dmjd = np.append(s1dmjd, hdrraw['MJD-OBS'])
        berv = np.append(berv, barycorr.value)  # in km.s.
        airmass = np.append(airmass, airmass_calc)

        # Read in the detector data
        hdu = fits.open(inpath / filelist[rawfiles_infileidx[rawfile]])
        chip1data = copy.deepcopy(hdu[1].data)
        chip2data = copy.deepcopy(hdu[2].data)
        chip3data = copy.deepcopy(hdu[3].data)
        hdu.close()
        del hdu[1].data
        del hdu[2].data
        del hdu[3].data

        npx = np.append(npx, 2048)
        if (len(chip1data)!=2048 or len(chip2data)!=2048 or len(chip3data)!=2048):
            raise("one of the crires chips were unexpectedly not 2048 pixels...")
        if (len(chip1data[0])%3!=0 or len(chip2data[0])%3!=0 or len(chip3data[0])%3!=0):
            raise("one of the crires chips did not have wave, flux and err for all orders (multiplet of 3)")
        chip1orders = int(len(chip1data[0])/3)
        chip2orders = int(len(chip2data[0])/3)
        chip3orders = int(len(chip3data[0])/3)
        norders = np.append(norders, chip1orders+chip2orders+chip3orders)

        # mapping starting wavelenghts to sort
        chiplookup = {}
        chiplookup_end = {}
        for idx in range(chip1orders):
            chiplookup["1_"+str(idx)] = chip1data[0][idx*3+2]
            chiplookup_end["1_" + str(idx)] = chip1data[2047][idx * 3 + 2]
        for idx in range(chip3orders):
            chiplookup["2_" + str(idx)] = chip2data[0][idx * 3 + 2]
            chiplookup_end["2_" + str(idx)] = chip2data[2047][idx * 3 + 2]
        for idx in range(chip3orders):
            chiplookup["3_" + str(idx)] = chip3data[0][idx * 3 + 2]
            chiplookup_end["3_" + str(idx)] = chip3data[2047][idx * 3 + 2]
        chiplookup_sorted = dict(sorted(chiplookup.items(), key=lambda item: item[1])).keys()

        # disentangle tuples...
        fluxes = []
        errs = []
        waves = []
        for chiplk in chiplookup_sorted:
            (chipid,orderid_str) = chiplk.split("_")
            orderid = int(orderid_str)
            loopflux = []
            looperrs = []
            loopwave = []
            if chipid == '1':
                for pixelidx in range(2048):
                    loopflux.append(chip1data[pixelidx][orderid * 3])
                    looperrs.append(chip1data[pixelidx][orderid * 3 + 1])
                    loopwave.append(ops.vactoair(chip1data[pixelidx][orderid * 3 + 2]))
            elif chipid == '2':
                for pixelidx in range(2048):
                    loopflux.append(chip2data[pixelidx][orderid * 3])
                    looperrs.append(chip2data[pixelidx][orderid * 3 + 1])
                    loopwave.append(ops.vactoair(chip2data[pixelidx][orderid * 3 + 2]))
            elif chipid == '3':
                for pixelidx in range(2048):
                    loopflux.append(chip3data[pixelidx][orderid * 3])
                    looperrs.append(chip3data[pixelidx][orderid * 3 + 1])
                    loopwave.append(ops.vactoair(chip3data[pixelidx][orderid * 3 + 2]))
            fluxes.append(np.array(loopflux))
            errs.append(np.array(looperrs))
            waves.append(np.array(loopwave))

        wavedata1d = []
        fluxdata1d = []
        for orderid in range(len(waves)):
            for wave in waves[orderid]:
                wavedata1d.append(wave)
            for flux in fluxes[orderid]:
                fluxdata1d.append(flux)

        out_wave.append(np.array(waves))
        out_e2ds.append(np.array(fluxes))
        out_wave1d.append(np.array(wavedata1d)*10)#This needs to be angstroms for archaic reasons.
        out_s1d.append(np.array(fluxdata1d))

    output = {'wave': out_wave, 'e2ds': out_e2ds, 'header': header,
              'wave1d': out_wave1d, 's1d': out_s1d, 's1dhdr': header,
              'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 's1dmjd': s1dmjd,
              'npx': npx, 'norders': norders, 'berv': berv, 'airmass': airmass}
    return (output)








def read_crires(inpath,filelist,read_s1d=True,nod='both'):
    import os
    import glob
    import copy
    from pathlib import Path
    import pathlib
    import numpy as np
    from astropy.time import Time
    from astropy import units as u
    from astropy.coordinates import SkyCoord, EarthLocation

    # The following variables define lists in which all the necessary data will be stored.
    framename = []
    header = []
    obstype = []
    texp = np.array([])
    date = []
    mjd = np.array([])
    s1dmjd = np.array([])
    airmass = np.array([])
    berv = np.array([])

    npx = np.array([])
    norders = np.array([])
    out_wave = []
    out_e2ds = []
    out_wave1d = []
    out_s1d = []


    for i in range(len(filelist)):
        nod_trigger = False#We skip this file unless this becomes true:
        if nod =='A' or nod =='B':#If we only want one nod:
            if filelist[i].startswith(f'cr2res_obs_nodding_extracted{nod}'):
                nod_trigger = True
        else:#If we want both nods at the same time:
            if filelist[i].startswith('cr2res_obs_nodding_extractedA') or filelist[i].startswith('cr2res_obs_nodding_extractedB.fits'):
                nod_trigger = True
        reducedfile = inpath / filelist[i]

        if reducedfile.is_file() and nod_trigger:
            print(filelist[i])
            with fits.open(reducedfile) as hdu:
                hdrraw = hdu[0].header
                airmass_calc = 0.5 * (float(hdrraw['HIERARCH ESO TEL AIRM START']) + float(hdrraw['HIERARCH ESO TEL AIRM END']))
                observatory = EarthLocation.from_geodetic(lat=hdrraw['HIERARCH ESO TEL GEOLAT'] * u.deg,
                                                        lon=hdrraw['HIERARCH ESO TEL GEOLON'] * u.deg,
                                                        height=hdrraw['HIERARCH ESO TEL GEOELEV'] * u.m)
                sc = SkyCoord(ra=hdrraw['RA'] * u.deg,dec=hdrraw['DEC'] * u.deg)
                barycorr = sc.radial_velocity_correction(obstime=Time(hdrraw['MJD-OBS'],
                                                format='mjd'),location=observatory).to(u.km / u.s)
                hdrraw['HIERARCH ESO QC BERV'] = barycorr.value
                hdrraw['HIERARCH ESO QC AIRMASS'] = airmass_calc
                framename.append(str(reducedfile))
                header.append(hdrraw)
                obstype.append('SCIENCE')
                texp = np.append(texp, hdrraw['HIERARCH ESO DET SEQ1 EXPTIME'])
                date.append(hdrraw['DATE-OBS'])
                mjd = np.append(mjd, hdrraw['MJD-OBS'])
                s1dmjd = np.append(s1dmjd, hdrraw['MJD-OBS'])
                berv = np.append(berv, barycorr.value)  # in km.s.
                airmass = np.append(airmass, airmass_calc)
                # Read in the detector data
                chip1data = copy.deepcopy(hdu[1].data)
                chip2data = copy.deepcopy(hdu[2].data)
                chip3data = copy.deepcopy(hdu[3].data)

            npx = np.append(npx, 2048)
            if (len(chip1data)!=2048 or len(chip2data)!=2048 or len(chip3data)!=2048):
                raise Exception("one of the crires chips were unexpectedly not 2048 pixels...")
            if (len(chip1data[0])%3!=0 or len(chip2data[0])%3!=0 or len(chip3data[0])%3!=0):
                raise Exception("one of the crires chips did not have wave, flux and err for all orders (multiplet of 3)")
            chip1orders = int(len(chip1data[0])/3)
            chip2orders = int(len(chip2data[0])/3)
            chip3orders = int(len(chip3data[0])/3)
            norders = np.append(norders, chip1orders+chip2orders+chip3orders)

            # mapping starting wavelenghts to sort
            chiplookup = {}
            chiplookup_end = {}
            for idx in range(chip1orders):
                chiplookup["1_"+str(idx)] = chip1data[0][idx*3+2]
                chiplookup_end["1_" + str(idx)] = chip1data[2047][idx * 3 + 2]
            for idx in range(chip3orders):
                chiplookup["2_" + str(idx)] = chip2data[0][idx * 3 + 2]
                chiplookup_end["2_" + str(idx)] = chip2data[2047][idx * 3 + 2]
            for idx in range(chip3orders):
                chiplookup["3_" + str(idx)] = chip3data[0][idx * 3 + 2]
                chiplookup_end["3_" + str(idx)] = chip3data[2047][idx * 3 + 2]
            chiplookup_sorted = dict(sorted(chiplookup.items(), key=lambda item: item[1])).keys()

            # disentangle tuples...
            fluxes = []
            errs = []
            waves = []
            for chiplk in chiplookup_sorted:
                (chipid,orderid_str) = chiplk.split("_")
                orderid = int(orderid_str)
                loopflux = []
                looperrs = []
                loopwave = []
                if chipid == '1':
                    for pixelidx in range(2048):
                        loopflux.append(chip1data[pixelidx][orderid * 3])
                        looperrs.append(chip1data[pixelidx][orderid * 3 + 1])
                        loopwave.append(ops.vactoair(chip1data[pixelidx][orderid * 3 + 2]))
                elif chipid == '2':
                    for pixelidx in range(2048):
                        loopflux.append(chip2data[pixelidx][orderid * 3])
                        looperrs.append(chip2data[pixelidx][orderid * 3 + 1])
                        loopwave.append(ops.vactoair(chip2data[pixelidx][orderid * 3 + 2]))
                elif chipid == '3':
                    for pixelidx in range(2048):
                        loopflux.append(chip3data[pixelidx][orderid * 3])
                        looperrs.append(chip3data[pixelidx][orderid * 3 + 1])
                        loopwave.append(ops.vactoair(chip3data[pixelidx][orderid * 3 + 2]))
                fluxes.append(np.array(loopflux))
                errs.append(np.array(looperrs))
                waves.append(np.array(loopwave))

            wavedata1d = []
            fluxdata1d = []
            for orderid in range(len(waves)):
                for wave in waves[orderid]:
                    wavedata1d.append(wave)
                for flux in fluxes[orderid]:
                    fluxdata1d.append(flux)

            out_wave.append(np.array(waves))
            out_e2ds.append(np.array(fluxes))
            out_wave1d.append(np.array(wavedata1d)*10)#This needs to be angstroms for archaic reasons.
            out_s1d.append(np.array(fluxdata1d))

    output = {'wave': out_wave, 'e2ds': out_e2ds, 'header': header,'wave1d': out_wave1d,
                's1d': out_s1d, 's1dhdr': header,'mjd': mjd, 'date': date, 'texp': texp,
                'obstype': obstype, 'framename': framename, 's1dmjd': s1dmjd,
                'npx': npx, 'norders': norders, 'berv': berv, 'airmass': airmass}
    return (output)





def read_hires_makee(inpath,filelist,construct_s1d=True,N_CCD=3):
    """
    This reads a folder of HIRES data reduced by MIKEE. This is an experimental module based on one
    set of transit observations, courtesy J. Teske.

    Assumptions:
    1) There are 3 CCDs The final digit in the filename specifies the CCD number as _n.fits.
    2) There are non-reinterpolated extracted 2D order files.
    3) All CCDs have the same number of pixels, and those are all extracted (or if cropped, equally)
    for all 3 CCDs.
    4) The data are berv-corrected, so this berv-correction is undone.
    5) There exists no merged, stitched 1D spectrum, so one is constructed crudely by concatenating
    the individual spectral orders (which are fairly flat, so that is nice.)"""

    from tayph.system_parameters import calculateberv

    #The following variables define lists in which all the necessary data will be stored.
    framename=[]
    header=[]
    s1dhdr=[]
    obstype=[]
    texp=np.array([])
    date=[]
    mjd=np.array([])
    s1dmjd=np.array([])
    npx=np.array([])
    norders=np.array([])
    e2ds=[]
    s1d=[]
    wave1d=[]
    airmass=np.array([])
    berv=np.array([])
    wave=[]
    slice_val = 23
    print("hello")

    catkeyword = 'OBSTYPE'
    #bervkeyword = 'HELIOVEL'
    thfilekeyword = 'ARCSPFIL'

    # wavefile_used = []
    for i in range(len(filelist)):
        if filelist[i].startswith('Flux-') and filelist[i].endswith('_1.fits'):

            #We need to read from 3 CCDs.
            #These *should* all have the same number of extracted pixels, i.e. the width of the
            #detector.
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath/filelist[i])
            data1 = copy.deepcopy(hdul[0].data)
            data1 = data1[:slice_val]
            hdr1 = hdul[0].header
            hdul.close()
            del hdul[0].data
            hdul = fits.open(inpath/filelist[i].replace('_1.fits','_2.fits'))
            data2 = copy.deepcopy(hdul[0].data)
            hdr2 = hdul[0].header
            hdul.close()
            del hdul[0].data
            hdul = fits.open(inpath/filelist[i].replace('_1.fits','_3.fits'))
            data3 = copy.deepcopy(hdul[0].data)
            hdr3 = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr1[catkeyword].startswith('Object'):
                framename.append(filelist[i])
                header.append(hdr1)
                obstype.append('SCIENCE')
                texp=np.append(texp,float(hdr1['EXPOSURE']))
                date.append(hdr1['DATE-OBS']+'T'+str(hdr1['UTC']))#.replace(':','-'))
                mjd=np.append(mjd,float(hdr1['MJD']))
                npx=np.append(npx,int(hdr1['NAXIS1']))
                norders=np.append(norders,len(data1)+int(hdr2['NAXIS2'])+
                int(hdr3['NAXIS2']))

                berv_i= calculateberv(hdr1["DATE"],[hdr1['LATITUDE'],hdr1['LONGITUD'],4145],hdr1["RA2000"],
                hdr1["DEC2000"],"HIRES-MAKEE")
                #print(yest)


                berv=np.append(berv,berv_i)
                airmass=np.append(airmass,float(hdr1['AIRMASS']))
                merged_e2ds = np.vstack((data1,data2,data3))
                e2ds.append(merged_e2ds)#Stack them all in a massive E2DS matrix.


                wavedata1=ops.vactoair(ut.read_wave_from_makee_header(hdr1,slice_val)/10.0)#convert to nm.
                wavedata2=ops.vactoair(ut.read_wave_from_makee_header(hdr2)/10.0)#and back to air.
                wavedata3=ops.vactoair(ut.read_wave_from_makee_header(hdr3)/10.0)

                merged_wave = np.vstack((wavedata1,wavedata2,wavedata3))
                gamma = (1.0-(berv_i*u.km/u.s/const.c).decompose().value)#BERV.
                gamma  = 1
                wave.append(merged_wave*gamma)#Remove BERV.


                if construct_s1d:
                    #We want to concatenate the orders while crudely ignoring overlap regions.
                    start_wl = [np.min(w) for w in merged_wave]
                    start_wl.append(np.inf)

                    wave_chunks = []
                    spec_chunks = []
                    for i in range(len(merged_wave)):
                        wlmin = start_wl[i]
                        wlmax = start_wl[i+1]
                        ww = merged_wave[i]#Wide, uncropped.
                        s = merged_e2ds[i][(ww>wlmin)&(ww<wlmax)]
                        w = merged_wave[i][(ww>wlmin)&(ww<wlmax)]

                        wave_chunks.append(w)
                        spec_chunks.append(s)
                    concatenated_spec = np.hstack(spec_chunks)
                    concatenated_wave = np.hstack(wave_chunks)
                    hdr1d = copy.deepcopy(hdr1)
                    s1d.append(concatenated_spec)
                    hdr1d['TELALT'] = float(hdr1d['EL'])
                    hdr1d['UTC'] = (float(hdr1d['MJD'])%1.0)*86400.0
                    del hdr1d['SLITSIZE']
                    del hdr1d['SLITLENG']
                    s1dhdr.append(hdr1d)

                    t = Time(hdr1d['DATE-OBS'], format='isot', scale='utc')
                    hdr1d['MJD'] = float(t.mjd)
                    zpt = Time(Time(int(Time(hdr1d['DATE-OBS']).mjd), format='mjd').isot)
                    hdr1d['DATE-OBS-S'] = (abs(zpt - t)).to_value('sec')

                    hdul.writeto(path + dir_list[i], overwrite=True)
                    s1dhdr.append(hdr1d)
                    s1dmjd=np.append(s1dmjd,float(hdr1d['MJD']))
                    berv1d = berv_i
                    wave1d.append(concatenated_wave*gamma*10)#Need to convert back to angstroms...
    output = {'wave':wave,'e2ds':e2ds,'header':header,'wave1d':wave1d,'s1d':s1d,'s1dhdr':s1dhdr,
    'mjd':mjd,'date':date,'texp':texp,'obstype':obstype,'framename':framename,'npx':npx,
    'norders':norders,'berv':berv,'airmass':airmass,'s1dmjd':s1dmjd}
    return(output)

def read_fies(inpath,filelist,mode,read_s1d=True):
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy import units as u
    from tayph.system_parameters import calculateberv

    catkeyword = 'IMAGECAT'
    #bervkeyword = 'HIERARCH TNG DRS BERV'
    Zstartkeyword = 'AIRMASS'
    Zendkeyword = 'AIRMASS'  # These are the same because FIES doesnt have start and end keywords.

    output = "Read FIES File"

    # The following variables define lists in which all the necessary data will be stored.
    framename = []
    header = []
    s1dhdr = []
    obstype = []
    texp = np.array([])
    date = []
    mjd = np.array([])
    s1dmjd = np.array([])
    npx = np.array([])
    norders = np.array([])
    e2ds = []
    blaze = []
    s1d = []
    wave1d = []
    airmass = np.array([])
    berv = np.array([])
    wave = []

    for i in range(len(filelist)):
        if filelist[i].endswith('10_wave.fits'):
            print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath / filelist[i])
            data = copy.deepcopy(hdul[0].data)
            #print(hdul.info())
            hdr = hdul[0].header
            hdul.close()
            del hdul[0].data
            if hdr[catkeyword] == 'SCIENCE':
                framename.append(filelist[i])
                header.append(hdr)
                obstype.append(hdr[catkeyword])
                texp = np.append(texp, hdr['EXPTIME'])
                date.append(hdr['DATE-OBS'])
                t = Time(hdr['DATE-OBS'], format='isot', scale='utc')
                mjd = np.append(mjd, t.mjd)#FIES Header doesn't provide mjd
                npx = np.append(npx, hdr['NAXIS1'])
                norders = np.append(norders, hdr['NAXIS2'])
                e2ds.append(data[0]/data[2]**2)#Using bandID1
                berv_i = sp.calculateberv(Time(hdr['DATE-OBS']),
                                          [hdr['OBSGEO-X'], hdr['OBSGEO-Y'], hdr['OBSGEO-Z']],
                                          hdr['RA'],
                                          hdr['DEC'],
                                          "FIES")
                berv = np.append(berv, berv_i)
                airmass = np.append(airmass, 0.5 * (hdr[Zstartkeyword] + hdr[Zendkeyword]))  # This is an approximation where we take the mean airmass.
                wavedata = ut.read_wave_from_e2ds_header(hdr, mode=mode)/10   # convert to nm.
                wave.append(wavedata)

                if read_s1d:
                    s1d_path = inpath / Path(str(filelist[i]).replace('10_wave.fits', '11_merge.fits'))
                    hdul = fits.open(s1d_path)
                    data_1d = copy.deepcopy(hdul[0].data)
                    hdr1d = hdul[0].header
                    hdul.close()
                    del hdul
                    s1d.append(data_1d)

                    s1dhdr.append(hdr1d)
                    t = Time(hdr['DATE-OBS'], format='isot', scale='utc')
                    s1dmjd = np.append(s1dmjd, t.mjd)

                    berv1d = sp.calculateberv(Time(hdr['DATE-OBS']),
                                              [hdr['OBSGEO-X'], hdr['OBSGEO-Y'], hdr['OBSGEO-Z']],
                                              hdr['RA'],
                                              hdr['DEC'],
                                              "FIES")
                    gamma = (1.0 - (berv1d * u.km / u.s / const.c).decompose().value)  # Doppler factor BERV.
                    gamma = 1
                    wave1d.append(np.linspace(hdr1d['CRVAL1'],hdr1d['CRVAL1'] + hdr1d['CDELT1']*len(data_1d),len(data_1d))*gamma)

    output = {'wave': wave, 'e2ds': e2ds, 'header': header, 'wave1d': wave1d, 's1d': s1d, 's1dhdr': s1dhdr,
              'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 'npx': npx,
              'norders': norders, 'berv': berv, 'airmass': airmass, 's1dmjd': s1dmjd}
    return (output)

def read_foces(inpath,filelist,mode,construct_s1d=True):

    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation
    from astropy import units as u
    from tayph.system_parameters import calculateberv
    import warnings
    warnings.filterwarnings("ignore")

    number_orders_foces = 84 #I had to manually measure this

    # The following variables define lists in which all the necessary data will be stored.
    framename = []
    header = []
    s1dhdr = []
    obstype = []
    texp = np.array([])
    date = []
    mjd = np.array([])
    s1dmjd = np.array([])
    npx = np.array([])
    norders = np.array([])
    e2ds = []
    blaze = []
    s1d = []
    wave1d = []
    airmass = np.array([])
    berv = np.array([])
    wave = []

    for i in range(len(filelist)):
        if filelist[i].endswith('SCI0_ods.fits'):
            #print(f'------{filelist[i]}', end="\r")
            hdul = fits.open(inpath / filelist[i])
            data = copy.deepcopy(hdul[1].data)
            # print(hdul.info())
            hdr = hdul[0].header
            hdr1 = hdul[1].header
            hdul.close()
            del hdul[0].data
            framename.append(filelist[i])
            header.append(hdr)
            obstype.append("SCIENCE")
            texp = np.append(texp, hdr['EXPOSURE'])
            date.append(hdr['DATE-OBS'])
            t = Time(hdr['DATE-OBS'], format='isot', scale='utc')
            mjd = np.append(mjd, t.mjd)  # FOCES Header doesn't provide mjd
            npx = np.append(npx, len(data[0][4]))
            norders = np.append(norders, number_orders_foces)

            collected_orders = []
            collected_waves = []
            collected_blaze = []
            collected_errors = []
            for j in range(0,number_orders_foces):
                #plt.plot(np.array(data[j][4]),np.array(data[j][12]))
                collected_orders.append(np.array(data[j][5]))
                collected_waves.append(np.array(data[j][4]))
                collected_errors.append(np.array(data[j][6]))
                collected_blaze.append(np.array(data[j][12]))
            #plt.show()
            collected_errors = 1 / np.sqrt(collected_errors)

            e2ds.append(np.array(collected_orders))

            berv_i = sp.calculateberv(mjd, [47.70364, 12.01206,1838],hdr['PERA2000'], hdr['PEDE2000'],mode=mode)
            berv = np.append(berv, berv_i)
            airmass = np.append(airmass, 1) #foces doesn't provide an airmass measurement, need to find a work around

            gamma = (1.0 - (berv_i * u.km / u.s / const.c).decompose().value)
            gamma = 1
            wave.append(np.array(collected_waves*gamma)/10)

            if construct_s1d:
                print("---Stitching 2D Eschelle Spectra Together.",end="\r")
                wave_1ds, data_1d = spec_stich_n_norm(np.array(collected_orders), np.array(collected_waves),
                                                     np.array(collected_blaze), np.array(collected_errors))

                s1d.append(data_1d)

                hdr1d = copy.deepcopy(hdr)
                #hdr1d['UTC'] = (float(hdr1d['MJD-OBS']) % 1.0) * 86400.0
                s1dhdr.append(hdr1d)
                s1dmjd = np.append(s1dmjd, t.mjd)

                wave1d.append(wave_1ds)


    if construct_s1d:
        output = {'wave': wave, 'e2ds': e2ds, 'header': header, 'wave1d': wave1d, 's1d': s1d, 's1dhdr': s1dhdr,
                  'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 'npx': npx,
                  'norders': norders, 'berv': berv, 'airmass': airmass, 's1dmjd': s1dmjd}
    else:
        output = {'wave': wave, 'e2ds': e2ds, 'header': header,
                  'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 'npx': npx,
                  'norders': norders, 'berv': berv, 'airmass': airmass}

    return (output)


def read_maroonx(inpath,filelist,construct_s1d=True):
    """
    This is a working read function for the Maroon-X spectrograph. It processes the reduced 2D echelle spectra, which
    are formatted as an .npz file, reads the relevant information, and constructs the files in a way that Tayph can
    interpret.

    To use this read function put your reduced data in a directory with the same name, without the .npz

    Parameters
    ----------
    inpath
    filelist
    construct_s1d

    Returns
    -------
    data files that can be cross-correlated
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.time import Time
    from tqdm import tqdm

    def process_channel_data_unified(data, channel):
        """
        Process the channel data ('blue' or 'red') to produce 1D spectra.

        Parameters:
            data (dict): The data dictionary containing flux, wavelengths, and variance for each channel.
            channel (str): The channel ('blue' or 'red') to be processed.
            blaze_model (function): The function to calculate the blaze model.
            spec_stich_n_norm (function): The function to stich and normalize the spectrum.
            adaptive_mad_filter_func (function): The function to apply adaptive MAD filtering.

        Returns:
            wave_1d (numpy array): The 1D wavelength array.
            spec_1d (numpy array): The 1D spectrum array.
        """

        # Extracting the flux and wavelength data for the channel
        flux = data[f'flux_{channel}']
        flux = np.transpose(flux, (1, 2, 0))

        wavelengths = data[f'wavelengths_{channel}']
        wavelengths = wavelengths.T

        var = data[f'f_var_{channel}']
        var = np.transpose(var, (1, 2, 0))

        wave_1ds = []
        spec_1ds = []
        print(f"---Constructing 1D for {channel} arm")
        for i in tqdm(range(0,len(flux))):

            flux_i = flux[i]
            flux_i = flux_i[::-1, :]

            wavelengths_i = wavelengths[::-1, :]

            var_i = var[i]
            var_i = var_i[::-1, :]

            std = np.sqrt(var_i)
            std = std / np.sum(std)

            # Applying blaze model and normalisation
            blaze = blaze_model(flux_i)
            normalised_spec = flux_i / blaze
            normalised_spec = normalised_spec / blaze_model(normalised_spec, window=25)
            ones_array = np.ones_like(flux_i)

            # Stiching and normalising the 1D spectrum
            wave_1d, spec_1d = spec_stich_n_norm(normalised_spec, wavelengths_i, ones_array, std)
            wave_1d = wave_1d*10
            wave_1ds.append(wave_1d)
            spec_1ds.append(spec_1d)

        return wave_1ds, spec_1ds
    # Load the data

    inpath = inpath.as_posix()
    data_file = inpath.split("/")
    data_file = data_file[-1]
    file_name = inpath + "/" + data_file + ".npz"

    data = np.load(file_name)

    flux_red = data['flux_red']
    flux_blue = data['flux_blue']

    wavelengths_red = ops.vactoair(data['wavelengths_red'])
    wavelengths_blue = ops.vactoair(data['wavelengths_blue'])
    dates = np.datetime_as_string(data['time'])

    berv_red = data['brv_red']
    berv_blue = data['brv_blue']
    airmass = data['airmass']

    # Taking the median between the berv estimates
    berv = np.median([berv_red,berv_blue],axis=0)/1000

    if construct_s1d:
        #Computer the 1D Spectrum
        wave_1d_blue, spec_1d_blue = process_channel_data_unified(data, 'blue')
        wave_1d_red, spec_1d_red = process_channel_data_unified(data, 'red')

        wave1ds = []
        s1ds = []
        for i in range(0,len(wave_1d_blue)):
            new_wave_1d_blue, new_wave_1d_red = make_same_shape(wave_1d_blue[i], wave_1d_red[i])
            new_spec_1d_blue, new_spec_1d_red = make_same_shape(spec_1d_blue[i], spec_1d_red[i])
            wave1d, s1d = average_overlapping_regions(new_wave_1d_blue, new_spec_1d_blue, new_wave_1d_red, new_spec_1d_red)
            wave1ds.append(wave1d)
            s1ds.append(s1d)

        wave1d = wave1ds
        s1d = s1ds


    # Determine the number of pixels for red and blue
    num_pixels_red = wavelengths_red.shape[0]
    num_pixels_blue = wavelengths_blue.shape[0]
    length_difference = num_pixels_red - num_pixels_blue

    # Initialize expanded blue wavelength and flux arrays
    expanded_wavelengths_blue = np.zeros((num_pixels_red, wavelengths_blue.shape[1]))
    padded_flux_blue = np.zeros((num_pixels_red, flux_blue.shape[1], flux_blue.shape[2]))

    # Insert NaNs for the first 'length_difference' elements in blue orders
    expanded_wavelengths_blue[:length_difference, :] = np.nan
    padded_flux_blue[:length_difference, :, :] = np.nan

    # Set the first 'length_difference' elements in blue orders to NaN
    expanded_wavelengths_blue[:length_difference, :] = np.nan
    padded_flux_blue[:length_difference, :, :] = np.nan

    # Vectorised version to expand blue wavelengths for all orders
    expanded_wavelengths_blue[length_difference:, :] = wavelengths_blue[:, :]

    # Vectorised version to pad flux with NaN for the expanded part
    padded_flux_blue[length_difference:, :, :] = flux_blue[:, :, :]

    # Concatenate red and blue arrays
    stacked_wavelengths = np.hstack((wavelengths_red, expanded_wavelengths_blue))
    stacked_flux = np.concatenate((flux_red, padded_flux_blue), axis=2)

    # Reshape the 2D stacked wavelength array into a 3D array
    reshaped_wavelengths = stacked_wavelengths[:, :, np.newaxis]

    # Replicate the 3D wavelength array across the third dimension to match the shape of the stacked flux array
    stacked_wavelengths_3d = np.tile(reshaped_wavelengths, (1, 1, stacked_flux.shape[1]))
    wave = stacked_wavelengths_3d

    # Transposing the stacked flux array to have the exposures as the first dimension of the array
    transposed_stacked_flux = np.transpose(stacked_flux, (1, 2, 0))
    e2ds = transposed_stacked_flux

    # Transposing the final stacked wavelengths 3D array to have the exposures as the first dimension of the array
    transposed_stacked_wavelengths_3d = np.transpose(stacked_wavelengths_3d, (2, 1, 0))
    wave = transposed_stacked_wavelengths_3d
    wave = wave[:, ::-1, :]
    wave = wave[:, :, ::-1]

    e2ds = transposed_stacked_flux
    e2ds = e2ds[:, ::-1, :]
    e2ds = e2ds[:, :, ::-1]

    # Creating the date values
    date = dates
    date = list(date)

    # Create MJD array
    t = Time(dates, format='isot', scale='utc')
    mjds = t.mjd
    mjd = mjds

    # Estimating the exposure times based on the time difference between succesive measurements, and using median
    exp_time = np.median(np.round((np.diff(mjds) * 24 * 60 * 60), 0))  # Converting to seconds
    exp_times = np.array(len(mjds) * [exp_time])
    texp = exp_times

    # Simulating header names for science
    obstype = np.array(len(dates) * ["SCIENCE"])
    obstype = list(obstype)

    # Removing the .npz value to the file name
    framename = file_name.split(".")
    framename = framename[0]

    # Number of exposures
    num_exposures = len(dates) * [len(dates)]

    # Creating a suffix array with values from _1 to _59
    suffixes = np.arange(1, len(dates) + 1)

    # Concatenating the base_string with the suffixes
    framename = np.core.defchararray.add(framename + '_', suffixes.astype(str))

    # Extracting the number of pixels
    data_shape = transposed_stacked_flux.shape
    npx = np.array(len(dates) * [data_shape[2]])

    # Extracting the number of orders
    data_shape = transposed_stacked_flux.shape
    norders = np.array(len(dates) * [data_shape[1]])

    #Extracting the airmass

    #The variable header is no longer used so we can just pass it as an empty list
    header = []

    # Create a new header
    s1dhdrs = []
    for i in range(0,len(mjd)):
        s1dhdr = fits.Header()
        dat = date[i]
        s1dhdr['MJD'] = float(mjd[i])
        s1dhdr['DATE-OBS'] = dates[i]
        s1dhdr['UTC'] = (float(s1dhdr['MJD']) % 1.0) * 86400.0
        zpt = Time(Time(int(Time(dat).mjd), format='mjd').isot)
        t = Time(dat, format='isot', scale='utc')
        s1dhdr['DATE-OBS-S'] = (abs(zpt - t)).to_value('sec')
        s1dhdrs.append(s1dhdr)

    if construct_s1d:
        output = {'wave': list(wave), 'e2ds': list(e2ds), 'header': header, 'wave1d': wave1d, 's1d': s1d, 's1dhdr': s1dhdrs,
                  'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 'npx': npx,
                  'norders': norders, 'berv': berv, 'airmass': airmass, 's1dmjd': mjd}

    else:
        output = {'wave': list(wave), 'e2ds': list(e2ds),
                  'mjd': mjd, 'date': date, 'texp': texp, 'obstype': obstype, 'framename': framename, 'npx': npx,
                  'norders': norders, 'berv': berv, 'airmass': airmass, 'header': header}

    return output