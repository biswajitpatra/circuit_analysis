from collections import Counter
import numpy as np
from math import pi,hypot
import math
import pygame


branch_to_findc=0


class Button(object):
    def __init__(self,colour,rect,command,**kwargs):
        self.rect = pygame.Rect(rect)
        self.colour=colour
        self.command = command
        self.clicked = False
        self.hovered = False
        self.hover_text = None
        self.clicked_text = None
        self.process_kwargs(kwargs)
        self.render_text()
 
    def process_kwargs(self,kwargs):
        settings = {
            "color"             : pygame.Color(self.colour),
            "text"              : None,
            "font"              : None, #pygame.font.Font(None,16),
            "call_on_release"   : True,
            "hover_color"       : None,
            "clicked_color"     : None,
            "font_color"        : pygame.Color("white"),
            "hover_font_color"  : None,
            "clicked_font_color": None,
            "click_sound"       : None,
            "hover_sound"       : None,
            'border_color'      : pygame.Color('black'),
            'border_hover_color': pygame.Color('yellow'),
            'disabled'          : False,
            'disabled_color'     : pygame.Color('grey'),
            'radius'            : 3,
        }
        for kwarg in kwargs:
            if kwarg in settings:
                settings[kwarg] = kwargs[kwarg]
            else:
                raise AttributeError("{} has no keyword: {}".format(self.__class__.__name__, kwarg))
        self.__dict__.update(settings)
 
    def render_text(self):
        if self.text:
            if self.hover_font_color:
                color = self.hover_font_color
                self.hover_text = self.font.render(self.text,True,color)
            if self.clicked_font_color:
                color = self.clicked_font_color
                self.clicked_text = self.font.render(self.text,True,color)
            self.text = self.font.render(self.text,True,self.font_color)
 
    def get_event(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.on_click(event)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.on_release(event)
 
    def on_click(self,event):
        if self.rect.collidepoint(event.pos):
            self.clicked = True
            if not self.call_on_release:
                self.function()
 
    def on_release(self,event):
        if self.clicked and self.call_on_release:
            #if user is still within button rect upon mouse release
            if self.rect.collidepoint(pygame.mouse.get_pos()):
                self.command()
        self.clicked = False
 
    def check_hover(self):
        if self.rect.collidepoint(pygame.mouse.get_pos()):
            if not self.hovered:
                self.hovered = True
                if self.hover_sound:
                    self.hover_sound.play()
        else:
            self.hovered = False
 
    def draw(self,surface):
        color = self.color
        text = self.text
        border = self.border_color
        self.check_hover()
        if not self.disabled:
            if self.clicked and self.clicked_color:
                color = self.clicked_color
                if self.clicked_font_color:
                    text = self.clicked_text
            elif self.hovered and self.hover_color:
                color = self.hover_color
                if self.hover_font_color:
                    text = self.hover_text
            if self.hovered and not self.clicked:
                border = self.border_hover_color
        else:
            color = self.disabled_color
         
        #if not self.rounded:
        #    surface.fill(border,self.rect)
        #    surface.fill(color,self.rect.inflate(-4,-4))
        #else:
        if self.radius:
            rad = self.radius
        else:
            rad = 0
        self.round_rect(surface, self.rect , border, rad, 1, color)
        if self.text:
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text,text_rect)
             
             
    def round_rect(self, surface, rect, color, rad=20, border=0, inside=(0,0,0,0)):
        rect = pygame.Rect(rect)
        zeroed_rect = rect.copy()
        zeroed_rect.topleft = 0,0
        image = pygame.Surface(rect.size).convert_alpha()
        image.fill((0,0,0,0))
        self._render_region(image, zeroed_rect, color, rad)
        if border:
            zeroed_rect.inflate_ip(-2*border, -2*border)
            self._render_region(image, zeroed_rect, inside, rad)
        surface.blit(image, rect)
 
 
    def _render_region(self, image, rect, color, rad):
        corners = rect.inflate(-2*rad, -2*rad)
        for attribute in ("topleft", "topright", "bottomleft", "bottomright"):
            pygame.draw.circle(image, color, getattr(corners,attribute), rad)
        image.fill(color, rect.inflate(-2*rad,0))
        image.fill(color, rect.inflate(0,-2*rad))
         
    def update(self):
        #for completeness
        pass
         
class Particle:
    
    def __init__(self, x, y, size):
        self.x = x
        self.px=x
        self.y = y
        self.py=y
        self.size = size
        self.colour = (0, 0, 255)
        self.thickness = 0
        
class wire:
    def __init__(self,p1,p2,length,character):
        self.p1=p1
        self.p2=p2
        self.length = length
        self.character=character

class Environment:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.springs = []
        self.wire=[]
        
        self.colour = (255,255,255)
    

    def addParticles(self, n=1, **kargs):
        
        for i in range(n):
            size = kargs.get('size',10)
            mass = kargs.get('mass',10)
            x=kargs.get('x')
            y=kargs.get('y')
            particle = Particle(x, y, size)
            self.particles.append(particle)

    def addwire(self,p1,p2,length,character):
        self.wire.append(wire(self.particles[p1],self.particles[p2],length,character))

    def findParticle(self, x, y):
        """ Returns any particle that occupies position x, y """
        
        for particle in self.particles:
            if math.hypot(particle.x - x, particle.y - y) <= particle.size:
                return particle
        return None



(width, height) = (1200,800)
pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('BEE MADE SIMPLE')

universe = Environment(width, height)
universe.colour = (255,255,255)

character=['r',0]
def print_on_press():
    global character,text
    character=['r',int(text)]
    text=''
def print_on_press1():
    global character,text
    character=['c',int(text)]
    text=''
def print_on_press2():
    global character,text
    character=['v',int(text)]
    text=''
def print_on_press3():
    global character,text
    character=['r',0]
    text=''
colvswo={"r":'red','c':'green','v':'blue'}
settings = {"clicked_font_color":(0,0,0),"hover_font_color":(205,195, 100),'font':pygame.font.Font(None,16),'font_color':(255,255,255),'border_color':(0,0,0)}
btn = Button(rect=(10,10,105,25),colour='red', command=print_on_press, text='RESISTANCE', **settings)
btn1 = Button(rect=(120,10,105,25),colour='green', command=print_on_press1, text='CURRENT source ', **settings)
btn2 = Button(rect=(230,10,105,25),colour='blue', command=print_on_press2, text='voltage ', **settings)
btn3= Button(rect=(340,10,105,25),colour= 'orange',command=print_on_press3, text='wire', **settings)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)
clock = pygame.time.Clock()
input_box = pygame.Rect(450, 10, 140, 32)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
colortext = color_inactive
active = False
text = ''
    
def createuni():
 global text
 global colortext
 global character
 global active
 t=f=True
 while t:
    screen.fill(universe.colour)
    mouse = pygame.mouse.get_pos()
    btn.draw(screen)
    btn1.draw(screen)
    btn2.draw(screen)
    btn3.draw(screen)
    # Render the current text.
    txt_surface = font.render(text, True, colortext)
    # Resize the box if the text is too long.
    width = max(200, txt_surface.get_width()+10)
    input_box.w = width
    # Blit the text.
    screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
    # Blit the input_box rect.
    pygame.draw.rect(screen, colortext, input_box, 2)
    for p in universe.particles:
                pygame.draw.circle(screen, (0,0,0), (int(p.x), int(p.y)), p.size, 0)
    for s in universe.wire:
              if s.character[1]==0:
                 pygame.draw.line(screen, pygame.Color('orange'), (int(s.p1.x), int(s.p1.y)), (int(s.p2.x), int(s.p2.y)),6)
              else:   
                 pygame.draw.line(screen, pygame.Color(colvswo[s.character[0]]), (int(s.p1.x), int(s.p1.y)), (int(s.p2.x), int(s.p2.y)),6)
    pygame.display.flip() 
    for event in pygame.event.get():
        btn.get_event(event)
        btn1.get_event(event)
        btn2.get_event(event)
        btn3.get_event(event)
        if btn.clicked==True or btn1.clicked==True or btn2.clicked==True or btn3.clicked==True:
            continue
        if event.type ==pygame.MOUSEBUTTONDOWN and event.button==1:
            # If the user clicked on the input_box rect.
            if input_box.collidepoint(event.pos):
                # Toggle the active variable.
                active = not active
                colortext = color_active if active else color_inactive
                continue
            else:
                active = False
                colortext = color_active if active else color_inactive
                text=''
            universe.addParticles(x=pygame.mouse.get_pos()[0],y=pygame.mouse.get_pos()[1])
        elif event.type ==pygame.MOUSEBUTTONDOWN and event.button==3:
            selected_particle1 = universe.findParticle(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
            if selected_particle1==None:
                     continue
            x1,y1=selected_particle1.x,selected_particle1.y
            f=True
            while f:
             screen.fill(universe.colour)
             for p in universe.particles:
                pygame.draw.circle(screen, p.colour, (int(p.x), int(p.y)), p.size, 0)
             for s in universe.wire:
              if s.character[1]==0:
                 pygame.draw.line(screen, pygame.Color('orange'), (int(s.p1.x), int(s.p1.y)), (int(s.p2.x), int(s.p2.y)),6)
              else:
                 pygame.draw.line(screen,pygame.Color(colvswo[s.character[0]]) , (int(s.p1.x), int(s.p1.y)), (int(s.p2.x), int(s.p2.y)),6)
             pygame.draw.line(screen,(0,0,0),(x1,y1),pygame.mouse.get_pos())
             pygame.display.flip() 
             for event in pygame.event.get():               
              if event.type ==pygame.MOUSEBUTTONDOWN and event.button==3:
                 selected_particle2 = universe.findParticle(pygame.mouse.get_pos()[0],pygame.mouse.get_pos()[1])
                 if selected_particle2==None:
                     continue
                 x2,y2=selected_particle2.x,selected_particle2.y
                 ## ON TRIAL BASIS
                 universe.addwire(universe.particles.index(selected_particle1),universe.particles.index(selected_particle2), hypot(x2-x1,y2-y1),character)
                 f=False
                  
        elif event.type==pygame.KEYDOWN:
            if event.key ==pygame.K_SPACE:
                pygame.quit()
                return
            text+=event.unicode
                #t=False
                #break
    #pygame.display.update()
    clock.tick(30)
    
                
createuni()

lines=[]
characters=[]

indexing=universe.wire
for x in indexing:
    characters+=[x.character]
    lines+=[(universe.particles.index(x.p1),universe.particles.index(x.p2))]

amatrix=[]
bmatrix=[]

no_of_points=0
for x in lines:
    if no_of_points <x[0]:
        no_of_points=x[0]
    elif no_of_points<x[1]:
        no_of_points=x[1]
no_of_points+=1

action=[0 for x in range(len(lines))]
actionp=[0 for x in range(no_of_points)]
routes=[]
routes_desc=[]
#loops=[]

def routesadd(addr):
    global routes
    global routes_desc
    #routes+=[x for x in addr]
    routes_descp=[x[2] for x in addr]
    if routes_desc==[]:
        routes+=[[x for x in addr]]
        routes_desc+=[[x[2] for x in addr]]
    for x in routes_desc:
        if Counter(x)==Counter(routes_descp):
            return
    routes+=[[x for x in addr]]
    routes_desc+=[[x[2] for x in addr]]

def routing(x,proute):
    global routes
    global action
    global actionp
    global lines
    for i in range(len(lines)):
        if x==lines[i][0] and action[i]==0 and actionp[lines[i][1]]==0:
            proute+=[[lines[i][0],lines[i][1],i]]
            actionp[lines[i][1]]=1
            action[i]=1
            if lines[i][1]==proute[0][0]:
                routesadd(proute)
                #routes+=[[x for x in proute]]
                proute.pop()
                actionp[lines[i][1]]=0
                action[i]=0
                continue
            routing(lines[i][1],proute)
            proute.pop()
            actionp[lines[i][1]]=0
            action[i]=0
        elif x==lines[i][1] and action[i]==0 and actionp[lines[i][0]]==0:
            proute+=[[lines[i][1],lines[i][0],i]]
            actionp[lines[i][0]]=1
            action[i]=1
            if lines[i][0]==proute[0][0]:
                routesadd(proute)
                #routes+=[[x for x in proute]]
                proute.pop()
                actionp[lines[i][0]]=0
                action[i]=0
                continue
            routing(lines[i][0],proute)
            proute.pop()
            actionp[lines[i][0]]=0
            action[i]=0
        if x==lines[i][0] and action[i]==0 and actionp[lines[i][1]]==1:
            proute+=[[lines[i][0],lines[i][1],i]]
            takev=0
            for x in proute:
                if x[0]==lines[i][1]:
                    takev=proute.index(x)
            prouteextra=[proute[x] for x in range(takev,len(proute))]
            routesadd(prouteextra)
            #routes+=[[x for x in prouteextra]]
            proute.pop()
        if x==lines[i][1] and action[i]==0 and actionp[lines[i][0]]==1:
            proute+=[[lines[i][1],lines[i][0],i]]
            takev=0
            for x in proute:
                if x[0]==lines[i][0]:
                    takev=proute.index(x)
            prouteextra=[proute[x] for x in range(takev,len(proute))]
            routesadd(prouteextra)
            #routes+=[[x for x in prouteextra]]
            proute.pop()
def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def getMatrixInverse(m):
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return cofactors           
         
def eqn_solver(am,bm):
    while(len(am)>len(am[0])):
        am.pop()
        bm.pop()
        
    nn=0
    while(len(am)<len(am[0])):
        nn+=1
        for x in range(len(am)):
            am[x].pop()
    #af=am
    while(getMatrixDeternminant(am)==0):
      nn+=1  
      for x in range(len(am)):
            am[x].pop()
      am.pop()
      bm.pop()
      #af = np.array(am)
   # df= np.array(bm)
    print(am,bm)
    #print(np.linalg.det(af))
    #ans =np.matrix.tolist(np.dot(np.linalg.inv(af),df))
    X,Y=getMatrixInverse(am),bm
    ans=[[sum(a*b for a,b in zip(X_row,Y_col)) for Y_col in zip(*Y)] for X_row in X]
    for x in range(nn):
        ans+=[[0]]
    return ans    

 
class equation:
    def __init__(self,loop_no=None):
        self.loops=loops[loop_no]
        self.cur_loop=[]
        #for x in range(len(self.loops)):    
        for x in self.loops:
            if (x[0],x[1])==lines[x[2]]:
               self.cur_loop+=[[cur_line[x[2]].val,characters[x[2]][0],-characters[x[2]][1]]]
            elif (x[1],x[0])==lines[x[2]]:
               self.cur_loop+=[[cur_line[x[2]].val,characters[x[2]][0],characters[x[2]][1]]]
        print(self.cur_loop)       
    def output(self):
        self.a_m=[0 for x in range(len(loops))]
        self.b_m=0
        for x in self.cur_loop:
             for y in range(len(loops)):
                 if str(y) in x[0]:
                    if x[1]=='r': 
                      if x[0][x[0].index(str(y))-1]=='-':
                         self.a_m[y]+=-x[2]
                      elif x[0][x[0].index(str(y))-1]=='+':
                         self.a_m[y]+=x[2]
                    elif x[1]=='v':
                         self.b_m-=x[2]
                         break
                         
                         
        x=[self.a_m,self.b_m]
        return x
        
def combiner(arr):
    global amatrix
    global bmatrix
    amatrix=[x[0] for x in arr]
    bmatrix=[[x[1]] for x in arr]
    
        
class current:
   def __init__(self,val=None):
       if val==None:
           self.val=['b']
           return
       self.val=str(val)
   def __add__(self,arr):
       if self.val==['b']:
           return ['+']+[arr.val]
       return self.val +['+'] +[arr.val]
   def __sub__(self,arr):
       if self.val==['b']:
           return ["-"]+[arr.val]
       return self.val +["-"] +[arr.val]
   def getresult(self):
       r=0
       for x in range(len(self.val)):
           if self.val[x]=='+':
               r+=answerf[int(self.val[x+1])][0]
           elif self.val[x]=='-':
               r-=answerf[int(self.val[x+1])][0]
       return r
               

routing(0,[])
print(routes)
loops=routes

vloops=[1 for x in loops]
for ln in range(len(loops)):
    for x in loops[ln]:
        if characters[x[2]][0]=='c':
            vloops[ln]=0
            
cur_varo=[current(x) for x in range(len(loops))]
print("equATIONS")
cur_line=[current() for x in range(len(lines))]

for x in range(len(lines)):
    for y in loops:
        if [lines[x][0],lines[x][1],x] in y:
            cur_line[x].val=cur_line[x]+cur_varo[loops.index(y)]
            ##print(cur_line[lines.index(x)].val)
        elif [lines[x][1],lines[x][0],x] in y:
            cur_line[x].val=cur_line[x]-cur_varo[loops.index(y)]
            ##print(cur_line[lines.index(x)].val)
 
for x in cur_line:
    print(x.val)
    print(' ')

print(vloops)    
equa=[equation(x).output() for x in range(len(loops)) if vloops[x]==1]
for x in range(len(characters)):
    if characters[x][0]=='c':
       tempar=[[0 for x in loops],characters[x][1]]
       for y in range(len(cur_line[x].val)):
               if cur_line[x].val[y]=='-':
                   tempar[0][int(cur_line[x].val[y+1])]=-1
               elif cur_line[x].val[y]=='+':
                   tempar[0][int(cur_line[x].val[y+1])]=+1
       equa=[tempar] +equa
       
print("PRINTING EQUA")        
print(equa)
combiner(equa)
print(amatrix,bmatrix)
answerf=eqn_solver(amatrix,bmatrix)
print(answerf)
   
print(cur_line[branch_to_findc].getresult())
