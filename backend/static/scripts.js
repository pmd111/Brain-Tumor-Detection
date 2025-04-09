$(document).ready(function () {
    $(".upload-btn").hover(function () {
        $(this).css("background", "#fff");
        $(this).css("color", "#111");
    }, function () {
        $(this).css("background", "#ffcc00");
        $(this).css("color", "#111");
    });

    $(".image-container img").hover(function () {
        $(this).css("transform", "scale(1.1)");
        $(this).css("box-shadow", "0px 0px 15px rgba(255, 255, 255, 0.5)");
    }, function () {
        $(this).css("transform", "scale(1.0)");
        $(this).css("box-shadow", "none");
    });
});
