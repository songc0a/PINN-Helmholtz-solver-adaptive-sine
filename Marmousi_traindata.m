clear all 
close all

load('broc.mat');
load('mar_v.mat')

n = size(v);
dx = 0.025; dz = 0.025;
nx = n(2); nz = n(1);

dx = 0.025; dz = 0.025;

h  = [dz dx];

src_x = 180; ns = length(src_x);
src_z = 3;

sx =  (src_x-1)*dx; 
sz = (src_z-1)*dz;

z  = [0:n(1)-1]'*h(1);
x  = [0:n(2)-1]*h(2);

figure;
pcolor(x,z,v);
shading interp
axis ij
colorbar; colormap(broc)
caxis([1.5 4]);
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
set(gca,'FontSize',14)


npmlz = 50; npmlx = npmlz;
Nz = nz + 2*npmlz;
Nx = nx + 2*npmlx;

v_e=extend2d(v,npmlz,npmlx,Nz,Nx);

Ps1 = vti_getP_H(n,npmlz,npmlx,src_z,src_x);
Ps1 = Ps1'*500;
Ps1 = reshape(full(Ps1),[nz+2*npmlz,nx+2*npmlx]);
Ps3 = imgaussfilt(Ps1,3);
Ps1 = sparse(Ps3(:));

fre = 3;
fm = 8; dt = 0.0005;
[zb,ff,nzb,T_zb]=subwavelet_ricker(dt,fm,0);
df1 = (ff(2) - ff(1));
k = round(fre/df1)+1;
sour = zb(k);


[o,d,n] = grid2odn(z,x);
n=[n,1];

nb = [npmlz  npmlx 0];
n  = n + 2*nb;

omega = 2*pi*fre;
A = Helm2D((omega)./v_e(:),o,d,n,nb);
Ps = Ps1*sour;
U  = A\Ps;

z1  = [0:Nz-1]'*h(1);
x1  = [0:Nx-1]*h(2);

[zz,xx] = ndgrid(z1,x1);

U_2D = reshape(full(U),[Nz,Nx]);

amp = 0.5;

figure
pcolor(x1,z1,real(U_2D));
shading interp
axis ij
colorbar; colormap(broc)
xlabel('Distance (km)','FontSize',12)
ylabel('Depth (km)','FontSize',12);
caxis([-amp amp]);
set(gca,'FontSize',14)


Lpmlz = npmlz*dz;
Lpmlx = npmlx*dx;

a0 = 0.5;
% a0 = 1.79;

gz = zeros(Nz,1);
for iz = 1:Nz;
    if iz<npmlz+1;
       gz(iz) = 2*pi*a0*fm*((npmlz-iz)*dz/Lpmlz)^2;
       
    else if iz>=Nz-npmlz+1;
            gz(iz) = 2*pi*a0*fm*((iz-(Nz-npmlz+1))*dz/Lpmlz)^2;
        else gz(iz) = 0;
        end
    end
end

sz = repmat(gz,1,Nx);
ez = 1 - 1i*sz/omega;

gx = zeros(1,Nx);
for ix = 1:Nx;
    if ix<npmlx+1;
        gx(ix) = 2*pi*a0*fm*((npmlx-ix)*dx/Lpmlx)^2;
    else if ix>=Nx-npmlx+1;
            gx(ix) = 2*pi*a0*fm*((ix-(Nx-npmlx+1))*dx/Lpmlx)^2;
        else gx(ix) = 0;
        end
    end
end

sx = repmat(gx,Nz,1);
ex = 1 - 1i*sx/omega;

A = ez./ex; B = ex./ez; C = ex.*ez;

x_star = xx(:);
z_star = zz(:);
m = 1./v_e(:).^2;
Ps = full(Ps(:));
A = A(:); B = B(:); C = C(:);
U_real = real(full(U));
U_imag = imag(full(U));

% save Marmousi_3Hz_singlesource_ps.mat U_real U_imag x_star z_star m Ps A B C


















