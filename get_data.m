function [XX] = get_data(fname)

load (fname);
[i,j] = size(datastart);


XX = cell(j,i);

for a = 1:j
   for b = 1 : i 
       XX{a,b} = data(datastart(b,a): dataend(b,a));  
   end
end

end