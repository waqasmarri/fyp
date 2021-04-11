$(document).ready(function () {
    $('#images').on('change', function (e) {
        e.preventDefault();
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/uploader',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            beforeSend: function () {
                $('#img-caps').html('Caption: Loading...');
                $('#output').addClass('respimg');
                $('#img-success').html('Image Selected');
                $('#file-upload-input').removeClass('invalid-image');
                $('#img-caps').removeClass('invalid-caps');
                
            },
            success: function (data) {
                console.log(data);
                $('#img-caps').html('Caption: ' + data.caption);
            },
            error(request, error) {
                $('#output').attr('src', "/static/NEWUI/images/all/2.jpg");
                $('#img-success').html('Please select a valid image');
                $('#img-caps').addClass('invalid-caps');
                $('#img-caps').html('Error! Please try again.')
                $('#file-upload-input').addClass('invalid-image');
            }
        });
    });

    $('#subscribe-button').on('click', function (e) {
        e.preventDefault();
        var li = $('.enteremail').val()
        $.ajax({
            type: 'POST',
            url: '/link',
            data: li,
            contentType: false,
            cache: false,
            processData: false,
            beforeSend: function () {
                $('#img-caps').html('Caption: Loading...');
                $('#output').addClass('respimg');
                $('#output').attr('src', 'static/loader1.svg');
                $('#img-caps').removeClass('invalid-caps');
                $('#subscribe-email').removeClass('invalid-image');
            },
            success: function (data) {
                console.log(data);
                $('#output').removeClass('loader')
                $('#output').attr('src', data.name);
                $('#img-caps').html('Caption: ' + data.caption);
            },
            error(request, error) {
                $('#output').addClass('respimg');
                $('#output').attr('src', "/static/NEWUI/images/all/2.jpg");
                $('#img-caps').addClass('invalid-caps');
                $('#img-caps').html('Error! Please try again.')
                $('#subscribe-email').addClass('invalid-image');
            }
        });
    });
});