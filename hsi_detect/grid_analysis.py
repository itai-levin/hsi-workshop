from hsi_detect.utils import *

from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from jupyter_dash import JupyterDash
from dash import Dash, dcc, html, Input, Output, callback
import time
import json
import pandas as pd
import matplotlib
import datetime
import os
from tqdm import tqdm
from hsi_detect.image import HyperspectralImage
from hsi_detect.spectrum import Spectrum
from skimage.measure import find_contours
import warnings

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['axes.linewidth']= 0.5
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.minor.width'] = 0.5
matplotlib.rcParams['ytick.minor.width'] = 0.5

class GridImage(HyperspectralImage):
    def __init__ (self, hsi_file,
                  plastic_ratio = 0.5, 
                  concentration_map=None, 
                  crop_coords = None):
        """
        Parameters
        ---------
        hsi_file (str): name of HSI image file
        plastic_ratio (float): ratio of absorbance at 420nm/700nm above which to remove pixels. 
                               This is useful to remove pixels that are from the plastic 96-well plates used for many experiments.
        concentration_map (str): name of .csv file containing concentrations for pellets in the same directory as the HSI image.
        crop_coords (tuple of tuples): Coordinates for cropping the image in the format ((Y_TL, X_TL), (Y_BR, X_BR))
        """
        super().__init__(hsi_file)
        self.dir = os.path.dirname(hsi_file)
        if crop_coords is not None:
            self.image = self.image[crop_coords[0][0]:crop_coords[1][0], crop_coords[0][1]:crop_coords[1][1], :]
        
        self.make_rgb()

        if concentration_map is not None:
            self.conc_map = pd.read_csv(os.path.join(self.dir, concentration_map), header=None)
        self.plastic_mask = ((self.image[:,:,get_closest_wl_ind(420, self.centers)] / self.image[:,:,get_closest_wl_ind(700, self.centers)])<plastic_ratio)[:,:,np.newaxis].astype(float)
        self.plastic_mask[self.plastic_mask==0] = np.nan
        self.reflection_mask = (np.argmax(self.image, axis=2) > get_closest_wl_ind(650, self.centers))[:,:,np.newaxis].astype(float)
        self.reflection_mask[self.reflection_mask==0] = np.nan
        self.narrow_band_abs = {} 
        
class GridAnalysis():
    
    def __init__(self, savedir, use_saved, 
                 ctrl_img_path, expt_img_paths, concentration_map,
                 ignore_plastic=True, ignore_reflection=True, plastic_ratio=0.5,
                 crop_coords = None
                ):
        """
        Class designed to facilitate analaysis on an image where the samples are organized in a grid
        
        Parameters
        ---------
        savedir (str): Directory to save outputs of analysis
        use_saved (bool): Whether to use previously saved image coordinates for analysis
        ctrl_img_path (str): Paths for image that contains the control pellets
        expt_img_paths (list[str]): List of paths that contain the pellets to be analyzed
        conc_map (str): Name of file containing concentration information for each pellet
        ignore_plastic (bool): If true, remove plastic pixels
        ignore_reflection (bool): If true, remove pixels of light glare reflection
        IP (str): ip address to use for interactive dash displays
        crop_coords: a list of list of lists of coordinates in the format [[[Y_tl1,X_tl1],[Y_br1,X_br1]],...,[[Y_tln,X_tln],[Y_brn,X_brn]]]
        """
        today = datetime.date.today()
        self.date_str = today.strftime('%d%b%Y')
        self.savedir = savedir
        self.use_saved = use_saved
        self.ctrl_img_dir = ctrl_img_path
        self.expt_img_dirs = expt_img_paths
        self.ignore_plastic = ignore_plastic
        self.ignore_reflection = ignore_reflection
        self.IP = get_local_ip() #Needed to display interactive windows on remote servers
        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)
        self.ctrl_img = GridImage(ctrl_img_path, plastic_ratio=plastic_ratio)
        
        if crop_coords is None:
            crop_coords = [None] * len(expt_img_paths)
        self.expt_imgs = [GridImage(p, concentration_map=concentration_map, 
                                          plastic_ratio=plastic_ratio, 
                                          crop_coords = crop_coords[i]) for i,p in enumerate(expt_img_paths)]
        self.params_file = os.path.join(self.savedir,'saved_params.json')
        if not os.path.exists(self.params_file) and self.use_saved:
            warnings.warn('Saved parameters not found. Ignoring `use_saved`.')
            self.use_saved = False
        print ('Parameters_file:', self.params_file)
        if not self.use_saved:
            self.params = {}
        
        else:
            self.params = json.load(open(self.params_file, 'r'))
            self.radii, self.tl,self.bl,self.tr,self.br = [self.params[k] for k in ['RADII', 'TL', 'BL', 'TR', 'BR']]
            self.ctrl_img.coord_grid = np.array([self.params['BG_CENTERS']])
            for img, r in zip(self.expt_imgs, self.radii):
                img.radius = r
            self.ctrl_img.radius = r
            print ("Loaded params file succesfully")

        self.num_blots_x = []
        self.num_blots_y = []
        for img in self.expt_imgs:
            s = img.conc_map.shape
            self.num_blots_x.append(s[1])
            self.num_blots_y.append(s[0])
    
    def get_grids (self):
        try:
            self.coord_grids = []
            for i, im in enumerate(self.expt_imgs):
                left_col = interpolate_points(self.tl[i],self.bl[i], self.num_blots_y[i])
                right_col = interpolate_points(self.tr[i],self.br[i], self.num_blots_y[i])

                grid = np.stack([interpolate_points(l,r,self.num_blots_x[i]) for l,r in zip(left_col, right_col)])
                self.coord_grids.append(grid)
                im.coord_grid = grid
        except NameError:
            raise ('Error: Check that coordinates of grid edges have been defined')
    
    def define_ctrl_coordinates_interactive(self):
        self.bg_centers = np.array([]).reshape(0, 2) 
        cmap = plt.get_cmap('tab20')
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = Dash(__name__, external_stylesheets=external_stylesheets)
        img_toshow = self.ctrl_img.rgb.astype(np.uint8)
        fig = px.imshow(img_toshow)
        app.layout = html.Div([ 
            html.Div([dcc.Graph(id='img',figure=fig)]),                      
            html.Div([dcc.Graph(id='spec-graph')])
                              ])
        def get_spectrum(selected_pixels):

            fig = px.scatter (x=[400], y=[0])
            for i, (pix_x, pix_y) in enumerate(selected_pixels):
                fig.add_trace(
                    go.Scatter(x=self.ctrl_img.centers, y=np.array(self.ctrl_img.image)[pix_y, pix_x, :])
                )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(type='linear')
            fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
            return fig

        @callback(
            Output('spec-graph', 'figure'),
            Output('img', 'figure'),
            Input('img', 'clickData'),
        )
        def update_spectrum(clickData):
            palette = {i:cmap(i) for i in range(20)}
            if clickData is not None or len(self.bg_centers)>0:
                x= int(clickData['points'][0]['x'])
                y= int(clickData['points'][0]['y'])
                self.bg_centers = np.concatenate([self.bg_centers,[(x,y)]]).astype(int)
                self.ctrl_img.coord_grid = np.array([self.bg_centers])
                fig = get_spectrum(self.bg_centers)

                img_fig = px.imshow(img_toshow)
                for i, (pix_x, pix_y) in enumerate(self.bg_centers):
                    img_fig.add_trace(
                        go.Scatter(x=[int(pix_x)], y=[int(pix_y)])
                    )
            else:
                fig = px.scatter(x=[0],y=[0])
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(type='linear')
                fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
                img_fig = px.imshow(img_toshow)

            return [fig, img_fig]

        if self.IP is not None:
            app.run(self.IP, port=np.random.choice(range(8300,8900)), debug=True)
        else:
            app.run(mode='inline', debug=True)
    
    def interactive_get_coords(self, img):
        print ('updated')
        selected_coords = np.array([]).reshape(0, 2) 
        cmap = plt.get_cmap('tab20')
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        img_toshow = img.rgb.astype(np.uint8)
        fig = px.imshow(img_toshow)
        app.layout = html.Div([ 
            html.Div([dcc.Graph(id='img',figure=fig)]),                      
            html.Div([dcc.Graph(id='spec-graph')])
                              ])
        def get_spectrum(selected_pixels):

            fig = px.scatter (x=[400], y=[0])
            for i, (pix_x, pix_y) in enumerate(selected_pixels):
                fig.add_trace(
                    go.Scatter(x=img.centers, y=img[pix_y, pix_x, :])
                )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(type='linear')
            fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
            return fig

        @callback(
            Output('spec-graph', 'figure'),
            Output('img', 'figure'),
            Input('img', 'clickData'),
        )
        def update_spectrum(clickData):
            global selected_coords
            palette = {i:cmap(i) for i in range(20)}
            if clickData is not None or len(selected_coords)>0:
                x= int(clickData['points'][0]['x'])
                y= int(clickData['points'][0]['y'])
                selected_coords = np.concatenate([selected_coords,[(x,y)]]).astype(int)
                fig = get_spectrum(selected_coords)

                img_fig = px.imshow(img_toshow)
                for i, (pix_x, pix_y) in enumerate(selected_coords):
                    img_fig.add_trace(
                        go.Scatter(x=[int(pix_x)], y=[int(pix_y)])
                    )
            else:
                fig = px.scatter(x=[0],y=[0])
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(type='linear')
                fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})
                img_fig = px.imshow(img_toshow)

            return [fig, img_fig]

        if self.IP is not None:
            app.run(self.IP, port=np.random.choice(range(8300,8900)), debug=True)
        else:
            app.run(mode='inline', debug=True)
            
        return selected_coords
    
    def define_pellet_coordinates_interactive (self):
        cmap = plt.get_cmap('tab20')
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
        figs = []
        for img in self.expt_imgs:
            figs.append(px.imshow(img.rgb.astype(np.uint8)))

        html_ls = [dcc.Graph(id=f'img_{i}',figure=fig) for i, fig in enumerate(figs)]
        text_ls = [html.H1(id=f'text_{i}', children='Select center of top left sample') for i in range(len(figs))]
        
        div_ls = []
        for h,t in zip(html_ls, text_ls):
            div_ls.append(t)
            div_ls.append(h)
        app.layout = html.Div(div_ls)

        COUNTS = [0 for _ in self.expt_imgs]
        self.tl = [(0,0) for _ in self.expt_imgs]
        self.tr = [(0,0) for _ in self.expt_imgs]
        self.br = [(0,0) for _ in self.expt_imgs]
        self.bl = [(0,0) for _ in self.expt_imgs]
        self.blot_edge = [(0,0) for _ in self.expt_imgs]
        self.radii = [0 for _ in self.expt_imgs]
        SELECTED = [[] for _ in self.expt_imgs]

        for idx, _ in enumerate(self.expt_imgs):
            @callback(
                Output(f'img_{idx}', 'figure'),
                Output(f'text_{idx}', 'children'),
                Input(f'img_{idx}', 'clickData'),
                Input(f'img_{idx}', 'id'),
            )
            def select_points(clickData, fig_id):
#                 global COUNTS
                i = int(fig_id.split('_')[1])
                palette = {i:cmap(i) for i in range(20)}
                if clickData is not None or len(SELECTED[i])>0:
                    x= int(clickData['points'][0]['x'])
                    y= int(clickData['points'][0]['y'])
                    SELECTED[i].append((x,y))
                    img_fig = px.imshow(self.expt_imgs[i].rgb.astype(np.uint8))
                    for pix_x, pix_y in SELECTED[i]:
                        img_fig.add_trace(
                            go.Scatter(x=[pix_x], y=[pix_y])
                        )
                    COUNTS[i] = COUNTS[i] % 5
                    if COUNTS[i] == 0:
                        self.tl[i] = (x,y)
                        text = 'Select center of top right sample'
                    elif COUNTS[i] == 1:
                        self.tr[i] = (x,y)
                        text = 'Select center of bottom right sample'
                    elif COUNTS[i] == 2:
                        self.br[i] = (x,y)
                        text = 'Select center of bottom left sample'
                    elif COUNTS[i] == 3:
                        self.bl[i] = (x,y)
                        text = 'Select EDGE of TOP RIGHT sample'
                    elif COUNTS[i] == 4:
                        self.blot_edge[i] = (x,y)
                        radius = np.sqrt(np.sum(np.square(np.array(self.tl[i])-np.array(self.blot_edge[i]))))
                        self.radii[i] = radius
                        self.expt_imgs[i].radius = radius
                        # assume control has same radius as (the last defined) experimental 
                        self.ctrl_img.radius = radius
                        text = 'Selection complete!'
                    COUNTS[i]+=1    


                else:
                    text = 'Select center of top left sample'
                    img_fig = px.imshow(self.expt_imgs[i].rgb.astype(np.uint8))

                return img_fig, text


        if self.IP is not None:
            app.run(self.IP, port=np.random.choice(range(8300,8900)), debug=True)
        else:
            print ('Stargin app')
            app.run(mode='inline', debug=True)
            
    def save_parameters(self):
        self.params['FILE_NAMES'] = self.expt_img_dirs + [self.ctrl_img_dir]
        self.params['NUM_BLOTS_Y'] = self.num_blots_y
        self.params['NUM_BLOTS_X'] = self.num_blots_x
        self.params['RADII'] = self.radii
        self.params['TL'] = self.tl
        self.params['BL'] = self.bl
        self.params['TR'] = self.tr
        self.params['BR'] = self.br
        self.params['BG_CENTERS'] = self.bg_centers.tolist()
        with open(self.params_file ,'w') as f:
            f.write(json.dumps(self.params))
            print ('Wrote params to', self.params_file )
    
    
    def _apply_fxn_to_grid (self, fxn, img, data = None, allow_mask=True):
        specs = np.zeros((img.coord_grid.shape[0],img.coord_grid.shape[1],len(img.centers)))
        if data is None:
            masked_img = img.image
        else:
            masked_img = data
            
        if self.ignore_plastic and allow_mask:
            masked_img = masked_img * np.reshape(img.plastic_mask, img.plastic_mask.shape[:len(masked_img.shape)])
        if self.ignore_reflection and allow_mask:
            masked_img = masked_img * np.reshape(img.reflection_mask, img.reflection_mask.shape[:len(masked_img.shape)])
            
        res = []
        for j in range(img.coord_grid.shape[0]):
            row = []
            for i in range(img.coord_grid.shape[1]):
                x = int(img.coord_grid[j,i,0])
                y = int(img.coord_grid[j,i,1])
                t = int(y-img.radius)
                b = int(y+img.radius)
                l = int(x-img.radius)
                r = int(x+img.radius)
                row.append(fxn(masked_img[t:b,l:r]))      
            res.append(row)
        return np.array(res)
    
    def _visualize_area_of_averaging (self, img_obj, background, rectangles=True, color='red', remove_background = True, **imshow_kwargs):
        if background=='rgb':
            masked_img = img_obj.rgb
            masked_img = masked_img / 255
        elif background=='hsi':
            masked_img = img_obj.scores
        else:
            raise ValueError(f'background={background} is not supported')            
        if self.ignore_plastic and remove_background:
            masked_img = masked_img * np.reshape(img_obj.plastic_mask, img_obj.plastic_mask.shape[:len(masked_img.shape)])
        if self.ignore_reflection and remove_background:
            masked_img = masked_img * np.reshape(img_obj.reflection_mask, img_obj.reflection_mask.shape[:len(masked_img.shape)])
        if background=='rgb':
            masked_img[np.isnan(masked_img)] = 1
        
        
        if rectangles:
            plt.figure(dpi=300)
            plt.imshow(masked_img, **imshow_kwargs)
            for j in range(img_obj.coord_grid.shape[0]):
                for i in range(img_obj.coord_grid.shape[1]):
                    x = int(img_obj.coord_grid[j,i,0])
                    y = int(img_obj.coord_grid[j,i,1])
                    t = int(y-img_obj.radius)
                    b = int(y+img_obj.radius)
                    l = int(x-img_obj.radius)
                    r = int(x+img_obj.radius)
                    rect = plt.Rectangle((l, t), r-l, b-t, fill=None, edgecolor=color)
                    plt.gca().add_patch(rect)
        else:
            contour_mask = img_obj.plastic_mask.copy()
            contour_mask[np.isnan(contour_mask)] = 0
            rectangle_tls = [(int(b-img_obj.radius), int(a-img_obj.radius)) for a, b  in [x for y in img_obj.coord_grid for x in y]]
            rectangle_brs = [(int(b+img_obj.radius), int(a+img_obj.radius)) for a, b  in [x for y in img_obj.coord_grid for x in y]]
            rectangle_mask = make_rectangle_mask(img_obj.image.shape[:2], zip(rectangle_tls, rectangle_brs))
            
            contours = find_contours(contour_mask[:,:,0] * rectangle_mask) 
            
            if remove_background:
                if background=='rgb':
                    masked_img = masked_img * contour_mask * rectangle_mask[:,:,np.newaxis]
                    masked_img[masked_img==0] = 1
                else:
                    contour_mask[contour_mask==0] = np.nan
                    rectangle_mask[rectangle_mask==0] = np.nan
                    masked_img = masked_img * contour_mask[:,:,0] * rectangle_mask
                
            
            plt.figure(dpi=300)
            plt.imshow(masked_img, **imshow_kwargs)
            
            for contour in contours:
                if np.max(np.max(contour[:,1])) - np.min(contour[:,1]) > img_obj.radius*0.8:
                    plt.plot(contour[:,1],contour[:,0], color=color, linewidth=0.8)

        
    def visualize_area_of_averaging(self, rectangles=True, color='red', background='rgb', remove_background=True, **imshow_kwargs):
        for idx, img_obj in enumerate(self.expt_imgs):        
            
            self._visualize_area_of_averaging(img_obj, background, rectangles=rectangles, color=color, remove_background=remove_background, **imshow_kwargs)
            
            plt.xticks([])
            plt.yticks([])
            plt.box(False)
            plt.savefig(f'{self.savedir}/{self.date_str }_rgb_img_with_marked_area_{idx}_{background}_rectangles={rectangles}.pdf', dpi=400)
            plt.show()

    def get_specs (self):
        """
        Returns the mean spectrum for each pellet as a matrix 
        """
        fxn = lambda x : np.nanmean(x, axis=(0,1))
        self.expt_specs = []
        for i, img in enumerate(self.expt_imgs):
            spec_grid = self._apply_fxn_to_grid(fxn, img)
            self.expt_specs.append(spec_grid)
            img.spec_grid = spec_grid
        ctrl_spec_grid = self._apply_fxn_to_grid(fxn, self.ctrl_img)
        self.ctrl_specs = ctrl_spec_grid
        self.ctrl_img.spec_grid = ctrl_spec_grid
        return [self.ctrl_specs] + self.expt_specs 
            
    def score_w_unmixing (self, scoring_model, threshold = 0, remove_plastic = False, 
                          reference_spectra = None, only_mask_for_endmembers = False
                          ):
        fxn = lambda x : np.nanmean(x)
        self.unmixing_scores = []
        for idx, img_obj in tqdm(enumerate(self.expt_imgs)):
            unfilt_img = img_obj.image
            norm_unfilt_img = unfilt_img / np.nanmax(unfilt_img, axis=2, keepdims=True)
            if remove_plastic and self.ignore_plastic and self.ignore_reflection:
                norm_img = norm_unfilt_img * np.reshape(img_obj.plastic_mask, img_obj.plastic_mask.shape[:len(norm_unfilt_img.shape)])
                norm_img = norm_img * np.reshape(img_obj.reflection_mask, img_obj.reflection_mask.shape[:len(norm_img.shape)])
            else:
                img = unfilt_img
                norm_img = norm_unfilt_img
                
            if reference_spectra is None:
                reference_spectrum = self.reference_spectrum
            else:
                reference_spectrum = reference_spectra[idx]
            
            # replace image object for the duration of the classification
            orig_image = img_obj.image
            img_obj.image = norm_img

            scoring_model.fit(img_obj, self.reference_spectrum)
            endmembers, clust_map = scoring_model.em_ls, scoring_model.clust_ls

            img_obj.clust_map = clust_map[0]
            scores = scoring_model.classify(reference_spectrum, threshold=threshold)
            img_obj.scores = scores
            img_obj.endmembers = endmembers[0]
            score_means = self._apply_fxn_to_grid(fxn, img_obj, img_obj.scores, 
                                                  allow_mask = not only_mask_for_endmembers)
            self.unmixing_scores.append(score_means)
            img_obj.scores_grid = score_means
            
            #restore variables
            img_obj.image_for_scoring = img_obj.image
            img_obj.image = orig_image
            
    def get_narrow_band_abs (self, wavelength, subtract_min=False):
        """
        Returns the mean of the infered absorbance at a designated wavelength for each pellet as a matrix
        """
        fxn = lambda x : 1-np.nanmean(x)
        abs_grid = []
        for i, img in enumerate(self.expt_imgs):
            narrow_band_img = img.image[:,:,get_closest_wl_ind(wavelength, img.centers)]
            abs_scores = self._apply_fxn_to_grid(fxn, img, narrow_band_img)
            if subtract_min:
                abs_scores = abs_scores - np.nanmin(abs_scores)
            img.narrow_band_abs[wavelength] = abs_scores
            abs_grid.append(abs_scores)
        return abs_grid
    
    def _flatten (self, list_of_lists):
        return np.array([x for y in list_of_lists for x in y])
    
    def define_reference_spectrum (self, use_ctrl_img = True, 
                                   norm_fxn = lambda x : x/np.max(x),
                                   path = None):
        
        collect_fxn = lambda x : x
        neg_specs = []
        pos_specs = []
        if use_ctrl_img:
            neg_specs = self._flatten(self._apply_fxn_to_grid(collect_fxn, self.ctrl_img))

        min_conc = np.min([np.nanmin(img.conc_map.values) for img in self.expt_imgs])
        max_conc = np.max([np.nanmax(img.conc_map.values) for img in self.expt_imgs])

        for img in self.expt_imgs:
            all_specs = self._apply_fxn_to_grid(collect_fxn, img)

            pos_coords = np.where(img.conc_map==max_conc)
            print (pos_coords)
            if len(self._flatten(self._flatten(all_specs[pos_coords]))):
                pos_specs.append(self._flatten(self._flatten(all_specs[pos_coords])))

            if not use_ctrl_img:
                neg_coords = np.where(img.conc_map==min_conc)
                neg_specs.append(all_specs[neg_coords])
        pos_specs = np.vstack(pos_specs)
        self.pos_spec = np.nanmean(pos_specs, axis=0)
        if not use_ctrl_img:
            neg_specs = np.vstack(neg_specs)
        self.neg_spec = np.nanmean(neg_specs, axis=(0,1,2))

        plt.plot(norm_fxn(self.pos_spec))
        plt.plot(norm_fxn(self.neg_spec))

        if path is None:
            reference_spectrum = norm_fxn(self.neg_spec) - norm_fxn(self.pos_spec)
            self.reference_spectrum = Spectrum(wavelengths=self.expt_imgs[0].centers, intensities=reference_spectrum)
        else:
            self.reference_spectrum = Spectrum(file_path=path)
            self.reference_spectrum.interpolate_spectrum(self.expt_imgs[0].centers)
        
        plt.figure()
        plt.plot(self.reference_spectrum.wavelengths, self.reference_spectrum.intensities)
        plt.show()
        self.max_abs = self.expt_imgs[0].centers[np.argmax(self.reference_spectrum.intensities)]
    
