---
layout: page
title: Experiments
order: 3
---

<div class="posts">
  {% for post in site.posts %}
    <article class="post">
      <h3><a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }}</a></h3>
      <div class="entry">
        {{ post.excerpt }}
      </div>
    </article>
  {% endfor %}
</div>
