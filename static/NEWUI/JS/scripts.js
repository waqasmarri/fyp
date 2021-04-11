//   all ------------------
function initSolonick() {
    "use strict";
    //   loader ------------------
    $(".pin").text("Loading");
    $(".loader-wrap").fadeOut(300, function () {
        $("#main").animate({
            opacity: "1"
        }, 300);
    });
    //   Background image ------------------
    var a = $(".bg");
    a.each(function (a) {
        if ($(this).attr("data-bg")) $(this).css("background-image", "url(" + $(this).data("bg") + ")");
    });

    //   parallax thumbnails position  ------------------
    $(".bg-parallax-module").each(function () {
        var tcp = $(this),
            dpl = tcp.data("position-left"),
            dpt = tcp.data("position-top");
        tcp.css({
            "top": dpt + "%"
        });
        tcp.css({
            "left": dpl + "%",
        });
    });
    $(".album-thumbnails div").each(function () {
        var dp2 = $(this).data("position-left2"),
            dpt2 = $(this).data("position-top2");
        $(this).css({
            "top": dpt2 + "%"
        });

        $(this).css({
            "left": dp2 + "%",
        });
    });
    $(".section-subtitle").fitText(0.85);


    //   slick  ------------------
    var sbp = $(".sp-cont-prev"),
        sbn = $(".sp-cont-next"),
        ccsi = $(".cur_carousel-slider-container"),
        scw = $(".slider-carousel-wrap"),
        fetcarCounter = $(".fet_pr-carousel-counter"),
        fpr = $('.fet_pr-carousel'),
        scs = $('.show-case-slider'),
        fcshinit = $('.fullscreen-slider'),
        ssn = $('.slider-nav'),
        ssnc = $('.slider-nav-counter'),
        fssc = $('.fullscreenslider-counter'),
        fshc = $('.fs-carousel');
    scs.on("init", function (event, slick) {
        fetcarCounter.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);

    });
    scs.slick({
        dots: true,
        infinite: true,
        speed: 600,
        slidesToShow: 1,
        centerMode: true,
        arrows: false,
        variableWidth: true,
    });
    scs.on("afterChange", function (event, slick, currentSlide) {
        var scsc = $(".show-case-item.slick-active").data("curtext");
        var $captproject = $(".single-project-title .caption");
        $captproject.text(scsc).shuffleLetters({});
        fetcarCounter.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    $('.single-slider').slick({
        infinite: true,
        slidesToShow: 1,
        dots: true,
        arrows: false,
        adaptiveHeight: true
    });
    fcshinit.on("init", function (event, slick) {
        fssc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    fcshinit.slick({
        infinite: true,
        slidesToShow: 1,
        dots: true,
        arrows: false,
        adaptiveHeight: false
    });
    fcshinit.on("afterChange", function (event, slick, currentSlide) {
        fssc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    $('.slideshow-container').slick({
        slidesToShow: 1,
        autoplay: true,
        autoplaySpeed: 4000,
        pauseOnHover: false,
        arrows: false,
        fade: true,
        cssEase: 'ease-in',
        infinite: true,
        speed: 1000
    });
    fshc.on("init", function (event, slick) {
        ssnc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    fshc.slick({
        infinite: true,
        slidesToShow: 3,
        dots: true,
        arrows: false,
        centerMode: false,
        responsive: [{
            breakpoint: 1224,
            settings: {
                slidesToShow: 2,
                centerMode: false,
            }
        },

        {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                centerMode: true,
            }
        }
        ]

    });
    fshc.on("afterChange", function (event, slick, currentSlide) {
        ssnc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    $(".fs-carousel-title h3 , .fs-carousel-link").on({
        mouseenter: function () {
            $(this).parents(".fs-carousel-item").find(".bg").addClass("hov-rot");
        },
        mouseleave: function () {
            $(this).parents(".fs-carousel-item").find(".bg").removeClass("hov-rot");
        }
    });
    $('.serv-carousel').slick({
        infinite: true,
        slidesToShow: 3,
        dots: true,
        arrows: false,
        centerMode: false,
        responsive: [{
            breakpoint: 1224,
            settings: {
                slidesToShow: 2,
                centerMode: false,
            }
        },

        {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                centerMode: true,
            }
        }
        ]

    });
    $('.half-slider-img').slick({
        arrows: false,
        infinite: true,
        fade: false,
        speed: 1000,
        vertical: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        asNavFor: '.slider-nav'
    });
    ssn.on("init", function (event, slick) {
        ssnc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    $('.slider-nav').slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        asNavFor: '.half-slider-img',
        dots: true,
        arrows: false,
        centerMode: false,
        focusOnSelect: false,
        fade: true,
    });
    ssn.on("afterChange", function (event, slick, currentSlide) {
        ssnc.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    $('.text-carousel').slick({
        infinite: true,
        slidesToShow: 3,
        slidesToScroll: 1,
        dots: true,
        arrows: false,
        centerPadding: "0",
        centerMode: true,
        responsive: [{
            breakpoint: 1224,
            settings: {
                slidesToShow: 2,
            }
        },

        {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                centerMode: true,
            }
        }
        ]

    });
    fpr.on("init", function (event, slick) {
        fetcarCounter.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    fpr.slick({
        infinite: true,
        slidesToShow: 4,
        slidesToScroll: 1,
        dots: true,
        arrows: false,
        slickCurrentSlide: 2,
        centerPadding: "0",
        centerMode: true,
        responsive: [{
            breakpoint: 1224,
            settings: {
                slidesToShow: 4,
                centerMode: false,
            }
        },

        {
            breakpoint: 1084,
            settings: {
                slidesToShow: 2,
                centerMode: true,
            }
        },

        {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                centerMode: true,
            }
        }
        ]

    });
    fpr.on("afterChange", function (event, slick, currentSlide) {
        fetcarCounter.text(Number(slick.currentSlide + 1) + ' / ' + slick.slideCount);
    });
    sbp.on("click", function () {
        $(this).closest(scw).find(ccsi).slick('slickPrev');
    });
    sbn.on("click", function () {
        $(this).closest(scw).find(ccsi).slick('slickNext');
    });

    //footer animation ------------------
    var n = $(".partcile-dec").data("parcount");
    $(".partcile-dec").jParticle({
        background: "rgba(255,255,255,0.0)",
        color: "rgba(255,255,255,0.081)",
        particlesNumber: n,
        particle: {
            speed: 20
        }
    });


    // Styles ------------------
    function csselem() {
        $(".height-emulator").css({
            height: $(".fixed-footer").outerHeight(true)
        });
        $(".show-case-slider .show-case-item").css({
            height: $(".show-case-slider").outerHeight(true)
        });
        $(".fullscreen-slider-item").css({
            height: $(".fullscreen-slider").outerHeight(true)
        });
        $(".half-slider-item").css({
            height: $(".half-slider-wrap").outerHeight(true)
        });
        $(".half-slider-img-item").css({
            height: $(".half-slider-img").outerHeight(true)
        });
        $(".hidden-info-wrap-bg").css({
            height: $(window).outerHeight(true) - 80 + "px"
        });
        $(".slideshow-item").css({
            height: $(".slideshow-container").outerHeight(true)
        });
        $(".fs-carousel-item").css({
            height: $(".fs-carousel").outerHeight(true)
        });
    }
    csselem();
    var $window = $(window);
    $window.resize(function () {
        csselem();
    });

    //   scroll to------------------

    $(".scroll-init  ul").singlePageNav({
        filter: ":not(.external)",
        updateHash: false,
        offset: 80,
        threshold: 120,
        speed: 800,
        currentClass: "act-scrlink"
    });
    $(".to-top").on("click", function (a) {
        a.preventDefault();
        $("html, body").animate({
            scrollTop: 0
        }, 800);
        return false;
    });
    $("<div class='to-top-letter'>t</div><div class='to-top-letter'>o</div><div class='to-top-letter'>p</div>").appendTo(".to-top span");

}
//   Parallax ------------------
function initparallax() {
    var a = {
        Android: function () {
            return navigator.userAgent.match(/Android/i);
        },
        BlackBerry: function () {
            return navigator.userAgent.match(/BlackBerry/i);
        },
        iOS: function () {
            return navigator.userAgent.match(/iPhone|iPad|iPod/i);
        },
        Opera: function () {
            return navigator.userAgent.match(/Opera Mini/i);
        },
        Windows: function () {
            return navigator.userAgent.match(/IEMobile/i);
        },
        any: function () {
            return a.Android() || a.BlackBerry() || a.iOS() || a.Opera() || a.Windows();
        }
    };
    trueMobile = a.any();
    if (null === trueMobile) {
        var b = new Scrollax();
        b.reload();
        b.init();
    }
    if (trueMobile) $(".bgvid , .background-vimeo , .background-youtube-wrapper ").remove();
}


//   Init All ------------------
$(function () {
    initparallax();
    initSolonick();
});

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#output')
                .attr('src', e.target.result);
        };
        reader.readAsDataURL(input.files[0]);
    }
}
function toggleNav(){
    $('.mob-nav').slideToggle('fast');
}
