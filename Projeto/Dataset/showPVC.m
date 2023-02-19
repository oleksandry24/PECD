%---------------------------- 
clear
close all
list={...
    'DPVC_116', 'DPVC_201', 'DPVC_221', 'DPVC_233', ...
    'DPVC_119', 'DPVC_203', 'DPVC_223', 'DPVC_106', ...
    'DPVC_200', 'DPVC_210', 'DPVC_228' };


ini=1;
W=25000;
fim=0;
for i=1:length(list)
    close all
    cmd=['load ' char(list(i)) ];
    eval(cmd);
    
    N=length(DAT.ecg);
    
    %------------------------------- windows
    for w=1:10000
        clc
        figure(1)
        ini=1+fim;
        fim=ini+W-1;
        %plot(1:N,DAT.ecg,'g', DAT.ind, DAT.ecg(DAT.ind),'b+')
        indices=find( DAT.ind<fim & DAT.ind>ini);
        plot(ini:fim,DAT.ecg(ini:fim),'g', DAT.ind(indices), DAT.ecg(DAT.ind(indices)),'b+')
        hold on
        pvc=max(ini,DAT.pvc.*DAT.ind);
        plot(pvc(indices),DAT.ecg(pvc(indices)),'ro','MarkerSize',4,'LineWidth',2 )
        title(list(i),'FontSize',22);
        xlabel(['Batimentos: ', num2str(length(DAT.ind))  '   '...
            'PVC :  ', num2str(sum(DAT.pvc))  ],'FontSize',18)
        zoom on
        DAT.ind(indices)'
        DAT.pvc(indices)'
        fim/length(DAT.ecg)
        get(0)
        pause
        close all
        
    end
    
end
