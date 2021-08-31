from django.shortcuts import render
from django.contrib import auth
from django.http import HttpResponseRedirect, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Post
import os
import numpy as np

dirname = os.path.dirname(__file__)

def login(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = auth.authenticate(username=username, password=password)

    if user is not None and user.is_active:
        auth.login(request, user)
        Post.objects.get_or_create(user=user)
        return HttpResponseRedirect('/')
    else:
        return render(request, 'aicore/login.html')

def logout(request):
    Post.objects.filter(user=request.user).delete()
    auth.logout(request)
    return HttpResponseRedirect('/')

def index(request):
    return render(request, 'aicore/index.html')

@csrf_exempt
def upload(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    user = auth.authenticate(username=username, password=password)
    try:
        post = Post.objects.get(user=user)
        return HttpResponse(post.publish(request))
    
    except Exception as exp:
        return HttpResponse(f"[Error]: {exp}")