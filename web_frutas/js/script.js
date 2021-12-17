
var loader = document.getElementsByClassName('loader')[0];
var container = document.getElementsByClassName('flex-container')[0];
document.body.style.backgroundImage = 'none';
document.body.style.overflow = "hidden";
container.style.visibility = 'hidden';
loader.id = 'show-loader';

//Animacion menu
var cont = 0;
var flag = 0;
window.addEventListener("scroll", function(event) {
    var top = this.scrollY,
        left =this.scrollX;
        if (top > 50 && flag == 0){
          var elem = document.getElementById('top-menu');
          elem.id = 'scroll-menu';
          flag=1;
        }
        else if (top <= 50 && flag == 1){
          var elem = document.getElementById('scroll-menu');
          elem.id = 'top-menu';
          flag=0;


        }
}, false);

//Animacion titulo
window.addEventListener("load", function(event) {
     loader.style.zIndez=2;
     document.body.style.backgroundImage = "linear-gradient(rgba(255,255,255,0.35), rgba(255,255,255,0.35)), url(\"../assets/images/background.jpg\")";
     setTimeout(() =>
     {
       loader.id = 'quit-loader';
       document.body.style.overflow = "visible";
       setTimeout(() =>
       {
         var elem1 = document.getElementById('subtitle-loading');
         var elem2 = document.getElementById('detector-loading');
         var elem3 = document.getElementById('fruit-loading');
         elem1.id = 'subtitle';
         elem2.id = 'detector';
         elem3.id = 'fruit';
       },
       1000);
     },1000);

  });
